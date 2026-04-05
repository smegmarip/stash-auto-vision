# Resource Manager Service

**GPU/VRAM Orchestration for Vision Services**

The Resource Manager coordinates GPU access across all vision services, ensuring efficient VRAM utilization and preventing out-of-memory errors when running multiple GPU-intensive services.

---

## Overview

### Problem Solved

With limited VRAM (16GB on RTX A4000), multiple services cannot run simultaneously:

- faces-service: ~4GB VRAM
- semantics-service: ~8GB peak (JoyCaption VLM, loaded/unloaded per job) + ~1.4GB (tag classifier, kept resident)

Resource Manager ensures only compatible services use GPU at once. The semantics service requests a GPU lease for the duration of JoyCaption captioning, then releases it after model unload. The tag classifier remains resident at ~1.4GB.

### Supported Compute Devices

| Device Type | Platform      | Detection                           |
| ----------- | ------------- | ----------------------------------- |
| **CUDA**    | NVIDIA GPUs   | Auto-detected via PyTorch           |
| **MPS**     | Apple Silicon | Auto-detected via PyTorch           |
| **ROCm**    | AMD GPUs      | Auto-detected via PyTorch HIP       |
| **CPU**     | Any           | Fallback with virtual memory budget |
| **Virtual** | Testing       | Simulated device for development    |

The service automatically detects the available compute device and manages resources accordingly. On Apple Silicon Macs, it estimates GPU memory based on unified memory (~75% of system RAM).

### Key Features

- **Lease-Based Allocation**: Services request/release GPU access
- **Priority Queue**: Higher priority jobs get GPU first
- **Heartbeat Timeout**: Abandoned leases automatically expire
- **Fair Scheduling**: FIFO within priority levels
- **Health Monitoring**: Tracks GPU status and queue

---

## API Endpoints

### Request GPU Access

```bash
POST /resources/gpu/request
```

**Request:**

```json
{
  "service_name": "semantics-service",
  "vram_required_mb": 8000,
  "priority": 3,
  "timeout_seconds": 300,
  "job_id": "semantics-abc123"
}
```

**Response (Granted):**

```json
{
  "request_id": "req-550e8400...",
  "granted": true,
  "lease_id": "lease-41d4a716...",
  "queue_position": null,
  "estimated_wait_seconds": null,
  "message": "GPU access granted"
}
```

**Response (Queued):**

```json
{
  "request_id": "req-550e8400...",
  "granted": false,
  "lease_id": null,
  "queue_position": 2,
  "estimated_wait_seconds": 120,
  "message": "Request queued at position 2"
}
```

### Check Request Status

```bash
GET /resources/gpu/request/{request_id}
```

**Response:**

```json
{
  "granted": false,
  "cancelled": false,
  "failed": false,
  "position": 1,
  "message": "Waiting in queue at position 1"
}
```

### Wait for GPU (Blocking)

```bash
POST /resources/gpu/wait/{request_id}
```

Blocks until GPU is granted or timeout expires.

### Release GPU

```bash
POST /resources/gpu/release
```

**Request:**

```json
{
  "lease_id": "lease-41d4a716..."
}
```

**Response:**

```json
{
  "released": true,
  "message": "GPU access released"
}
```

### Send Heartbeat

```bash
POST /resources/gpu/heartbeat
```

**Request:**

```json
{
  "lease_id": "lease-41d4a716..."
}
```

**Response:**

```json
{
  "success": true,
  "lease_valid": true,
  "expires_in_seconds": 540,
  "message": "Heartbeat received"
}
```

### Get GPU Status

```bash
GET /resources/gpu/status
```

**Response:**

```json
{
  "status": "in_use",
  "total_vram_mb": 16384,
  "available_vram_mb": 7884,
  "allocated_vram_mb": 8500,
  "active_leases": [
    {
      "lease_id": "lease-41d4a716...",
      "service_name": "semantics-service",
      "vram_allocated_mb": 8000,
      "granted_at": "2025-12-02T12:34:56Z",
      "expires_at": "2025-12-02T12:44:56Z",
      "last_heartbeat": "2025-12-02T12:35:56Z"
    }
  ],
  "queue_length": 1,
  "queue": [
    {
      "request_id": "req-789def...",
      "service_name": "faces-service",
      "vram_required_mb": 4000,
      "priority": 5,
      "position": 1
    }
  ]
}
```

### Health Check

```bash
GET /resources/health
```

---

## Resource Management

### Priority Levels

| Priority | Use Case                      |
| -------- | ----------------------------- |
| 1-3      | Critical/interactive requests |
| 4-6      | Normal batch processing       |
| 7-10     | Background/low-priority jobs  |

### Lease Lifecycle

1. **Request**: Service requests GPU with VRAM estimate
2. **Queue**: If GPU busy, request queued by priority
3. **Grant**: When VRAM available, lease created
4. **Heartbeat**: Service sends periodic heartbeats
5. **Release**: Service releases when done
6. **Expire**: Lease expires without heartbeat

### Timeout Configuration

| Parameter                   | Default | Description                          |
| --------------------------- | ------- | ------------------------------------ |
| `LEASE_DURATION_SECONDS`    | 600     | Max lease duration                   |
| `HEARTBEAT_TIMEOUT_SECONDS` | 60      | Time without heartbeat before expiry |

---

## Client Integration

### Python Client Usage

```python
from resource_client import ResourceManagerClient

client = ResourceManagerClient(
    resource_manager_url="http://resource-manager:5007",
    service_name="my-service"
)

# Request GPU
result = await client.request_gpu(
    vram_mb=8000,
    priority=5,
    timeout_seconds=300
)

if result["granted"]:
    # GPU access granted immediately
    lease_id = result["lease_id"]
else:
    # Wait for GPU
    result = await client.wait_for_gpu(
        result["request_id"],
        max_wait=300.0
    )

try:
    # Do GPU work...

    # Send heartbeats every 30 seconds
    await client.heartbeat()

finally:
    # Always release GPU
    await client.release_gpu()
```

### Heartbeat Pattern

```python
import asyncio

async def gpu_work_with_heartbeat(client, work_func):
    """Execute GPU work with periodic heartbeats"""

    # Start heartbeat task
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(30)
            await client.heartbeat()

    heartbeat_task = asyncio.create_task(heartbeat_loop())

    try:
        result = await work_func()
        return result
    finally:
        heartbeat_task.cancel()
        await client.release_gpu()
```

---

## Configuration

### Environment Variables

| Variable                    | Default | Description                                                |
| --------------------------- | ------- | ---------------------------------------------------------- |
| `TOTAL_VRAM_MB`             | 16384   | Memory budget in MB (used for CPU/virtual mode, or as cap) |
| `FORCE_DEVICE_TYPE`         | (auto)  | Force device type: `cuda`, `mps`, `cpu`, or `virtual`      |
| `LEASE_DURATION_SECONDS`    | 600     | Default lease duration                                     |
| `HEARTBEAT_TIMEOUT_SECONDS` | 60      | Heartbeat timeout                                          |
| `LOG_LEVEL`                 | INFO    | Log verbosity                                              |

### Device Detection Order

1. **CUDA** - Checked first on all platforms
2. **MPS** - Checked on macOS for Apple Silicon
3. **ROCm** - Checked via PyTorch HIP support
4. **CPU** - Fallback with virtual memory limit

Use `FORCE_DEVICE_TYPE=virtual` for testing without any GPU.

---

## Scheduling Algorithm

1. Requests sorted by (priority, timestamp)
2. When VRAM freed, highest priority request checked
3. If request fits in available VRAM, granted
4. Otherwise, next request checked
5. Expired requests removed from queue

### Example Scenario

```
Queue State:
  [P3] semantics-service: 8000MB  (JoyCaption VLM captioning phase)
  [P5] faces-service: 4000MB

Available: 16384MB

→ Grant semantics-service (8000MB, higher priority)
→ Available: 8384MB
→ Grant faces-service (4000MB)
→ Available: 4384MB

After semantics captioning completes:
→ semantics-service releases lease (VLM unloaded)
→ Available: 12384MB
→ Tag classifier remains resident (~1.4GB, no lease required)
```

---

## Monitoring

### Prometheus Metrics (Planned)

```
resource_manager_active_leases
resource_manager_queue_length
resource_manager_vram_allocated_mb
resource_manager_vram_available_mb
resource_manager_request_wait_seconds
```

### Log Analysis

```bash
# Watch GPU allocations
docker logs -f vision-resource-manager | grep -E "(granted|released|expired)"
```

---

## Troubleshooting

### GPU Always Busy

```bash
# Check active leases
curl http://localhost:5007/resources/gpu/status | jq .active_leases
```

If leases persist without heartbeats, they may be orphaned. Wait for timeout or restart service.

### Request Never Granted

- Check priority (lower number = higher priority)
- Check VRAM requirements fit total VRAM
- Check for stuck high-priority requests

### Memory Leak

If allocated VRAM doesn't match active services:

1. Check for orphaned leases
2. Restart resource-manager
3. Services will re-request GPU on next job

---

## Related Documentation

- [Semantics Service](SEMANTICS_SERVICE.md)
- [Faces Service](FACES_SERVICE.md)
- [Docker Architecture](DOCKER_ARCHITECTURE.md)
