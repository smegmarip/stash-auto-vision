# Vision API Documentation

**Service:** Vision API (Orchestrator)
**Port:** 5010
**Path:** `/vision`
**Status:** Phase 1 - Implemented
**Version:** 1.0.0

---

## Summary

The Vision API is the primary orchestrator and rollup service for the Stash Auto Vision platform. It coordinates all backend analysis modules (scenes, faces, semantics, objects) and provides a unified API for comprehensive video analysis. This service acts as the main entry point for clients and manages complex multi-module workflows.

As an orchestrator, the Vision API does not perform video processing directly. Instead, it coordinates calls to specialized backend services, manages job state across multiple modules, aggregates results, and handles error scenarios. The service supports both sequential processing (default) for resource-efficient operation and parallel processing (future) for maximum throughput.

The Vision API integrates with five backend services: frame-server (port 5001) for GPU-accelerated frame extraction, scenes-service (port 5002) for scene boundary detection, faces-service (port 5003) for face recognition, semantics-service (port 5004) for CLIP-based scene understanding (Phase 2), and objects-service (port 5005) for YOLO-World object detection (Phase 3). Health monitoring across all services ensures reliable operation.

In production testing, the Vision API successfully processed a 120-second video with both scenes and faces analysis in 13.97 seconds (sequential mode). The service provides comprehensive result aggregation, combining outputs from multiple backend services into a single unified response for easy client consumption.

---

## OpenAPI 3.0 Schema

```yaml
openapi: 3.0.3
info:
  title: Vision API - Orchestrator
  description: Main API gateway and multi-module coordinator for video analysis
  version: 1.0.0
servers:
  - url: http://vision-api:5010
    description: Internal Docker network
  - url: http://localhost:5010
    description: Local development

paths:
  /vision/analyze:
    post:
      summary: Submit comprehensive video analysis job
      description: |
        Submit a video for multi-module analysis. Enabled modules are processed
        according to the specified processing mode (sequential or parallel).
        Scene boundaries from scenes-service are passed to faces-service for
        optimized sampling.
      operationId: analyzeVideo
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AnalyzeRequest"
            examples:
              faces_and_scenes:
                summary: Faces and scenes analysis
                value:
                  video_path: /media/videos/scene.mp4
                  source_id: "12345"
                  enable_scenes: true
                  enable_faces: true
                  processing_mode: sequential
                  parameters:
                    scene_threshold: 27.0
                    face_min_confidence: 0.9
                    max_faces: 50
              faces_only:
                summary: Faces only
                value:
                  video_path: /media/videos/scene.mp4
                  source_id: "12345"
                  enable_faces: true
                  parameters:
                    face_min_confidence: 0.8
                    max_faces: 100
                    similarity_threshold: 0.6
      responses:
        "202":
          description: Analysis job accepted and queued
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeJobResponse"
        "400":
          description: Invalid request parameters
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "404":
          description: Video file not found
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "500":
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /vision/jobs/{job_id}/status:
    get:
      summary: Get analysis job status
      description: Poll job status and progress across all enabled modules
      operationId: getJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
          description: Job identifier returned from /vision/analyze
      responses:
        "200":
          description: Job status retrieved
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeJobStatus"
        "404":
          description: Job not found
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "500":
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /vision/jobs/{job_id}/results:
    get:
      summary: Get aggregated analysis results
      description: |
        Retrieve comprehensive results from all enabled modules. Only
        available when job status is "completed". Results include all
        module outputs aggregated into a single response.
      operationId: getJobResults
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        "200":
          description: Analysis results
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeResults"
        "404":
          description: Job not found
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "409":
          description: Job not completed yet
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"
        "500":
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /vision/jobs:
    get:
      summary: List all jobs across services
      description: |
        List all jobs from vision-api, faces-service, and scenes-service
        with optional filtering and pagination. Jobs are deduplicated
        by cache_key, preferring vision-level jobs.
      operationId: listJobs
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [queued, processing, completed, failed]
        - name: service
          in: query
          schema:
            type: string
            enum: [vision, faces, scenes]
        - name: source_id
          in: query
          schema:
            type: string
        - name: source
          in: query
          schema:
            type: string
        - name: start_date
          in: query
          schema:
            type: string
            format: date-time
        - name: end_date
          in: query
          schema:
            type: string
            format: date-time
        - name: include_results
          in: query
          schema:
            type: boolean
            default: false
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
      responses:
        "200":
          description: List of jobs
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListJobsResponse"

  /vision/health:
    get:
      summary: Service health check
      description: |
        Health status for Vision API and all downstream services.
        Returns detailed health information for scenes-service,
        faces-service, semantics-service, and objects-service.
      operationId: healthCheck
      responses:
        "200":
          description: Service healthy
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"
        "503":
          description: Service unhealthy
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

components:
  schemas:
    AnalyzeRequest:
      type: object
      required:
        - video_path
        - source_id
      properties:
        video_path:
          type: string
          description: Absolute path to video file on shared volume
          example: /media/videos/scene.mp4
        source_id:
          type: string
          description: Scene identifier for reference
          example: "12345"
        job_id:
          type: string
          format: uuid
          description: Optional custom job ID (generated if not provided)
        enable_scenes:
          type: boolean
          default: false
          description: Enable scene boundary detection
        enable_faces:
          type: boolean
          default: false
          description: Enable face recognition analysis
        enable_semantics:
          type: boolean
          default: false
          description: Enable semantic analysis (Phase 2 - currently stubbed)
        enable_objects:
          type: boolean
          default: false
          description: Enable object detection (Phase 3 - currently stubbed)
        processing_mode:
          type: string
          enum: [sequential, parallel]
          default: sequential
          description: |
            Processing mode - sequential (one module at a time) or
            parallel (all modules simultaneously). Sequential is default
            for resource efficiency.
        parameters:
          type: object
          description: Module-specific parameters
          properties:
            scene_detection_method:
              type: string
              enum: [content, threshold, adaptive]
              default: content
            scene_threshold:
              type: number
              default: 27.0
              description: Scene detection sensitivity (higher = fewer scenes)
            min_scene_length:
              type: number
              default: 0.6
              description: Minimum scene duration in seconds
            face_min_confidence:
              type: number
              default: 0.9
              minimum: 0
              maximum: 1
              description: Minimum face detection confidence
            max_faces:
              type: integer
              default: 50
              minimum: 1
              description: Maximum unique faces to detect
            face_sampling_interval:
              type: number
              default: 2.0
              description: Seconds between sampled frames for face detection
            enable_deduplication:
              type: boolean
              default: true
              description: Deduplicate faces by embedding similarity
            similarity_threshold:
              type: number
              default: 0.6
              minimum: 0
              maximum: 1
              description: Cosine similarity threshold for face deduplication
            detect_demographics:
              type: boolean
              default: true
              description: Extract age/gender/emotion from faces

    AnalyzeJobResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
          description: Job identifier for status polling
        status:
          type: string
          enum: [queued, processing, completed, failed]
        created_at:
          type: string
          format: date-time
        services_enabled:
          type: object
          properties:
            scenes:
              type: boolean
            faces:
              type: boolean
            semantics:
              type: boolean
            objects:
              type: boolean
        processing_mode:
          type: string
          enum: [sequential, parallel]

    AnalyzeJobStatus:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [queued, processing, completed, failed]
        progress:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Overall progress (0.0 to 1.0)
        processing_mode:
          type: string
          enum: [sequential, parallel]
        services:
          type: array
          description: Individual service statuses (future enhancement)
          items:
            type: object
        created_at:
          type: string
          format: date-time
        started_at:
          type: string
          format: date-time
          nullable: true
        completed_at:
          type: string
          format: date-time
          nullable: true
        error:
          type: string
          nullable: true
          description: Error message if status is "failed"

    AnalyzeResults:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        source_id:
          type: string
        status:
          type: string
          enum: [completed]
        scenes:
          type: object
          nullable: true
          description: Full scenes-service results (if enabled)
        faces:
          type: object
          nullable: true
          description: Full faces-service results (if enabled)
        semantics:
          type: object
          nullable: true
          description: Semantics results or stub response
        objects:
          type: object
          nullable: true
          description: Objects results or stub response
        metadata:
          type: object
          properties:
            processing_time_seconds:
              type: number
              format: float
            processing_mode:
              type: string
              enum: [sequential, parallel]
            services_used:
              type: object
              properties:
                scenes:
                  type: boolean
                faces:
                  type: boolean
                semantics:
                  type: boolean
                objects:
                  type: boolean

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        service:
          type: string
          example: vision-api
        version:
          type: string
          example: 1.0.0
        services:
          type: object
          description: Health status of all downstream services
          additionalProperties:
            type: object
            properties:
              status:
                type: string
              version:
                type: string
              error:
                type: string

    ErrorResponse:
      type: object
      properties:
        detail:
          type: string
          description: Error message

    ListJobsResponse:
      type: object
      properties:
        jobs:
          type: array
          items:
            $ref: "#/components/schemas/JobSummary"
        total:
          type: integer
          description: Total matching jobs before pagination
        limit:
          type: integer
        offset:
          type: integer

    JobSummary:
      type: object
      properties:
        job_id:
          type: string
        service:
          type: string
          enum: [vision, faces, scenes]
        status:
          type: string
          enum: [queued, processing, completed, failed]
        progress:
          type: number
        source:
          type: string
          nullable: true
        source_id:
          type: string
          nullable: true
        created_at:
          type: string
          format: date-time
        result_summary:
          type: object
          nullable: true
        results:
          type: object
          nullable: true
          description: Full results when include_results=true
```

---

## Functional Details

### Orchestration Architecture

The Vision API implements a hub-and-spoke architecture where it acts as the central coordinator for all backend services:

**Service Coordination:**

- Submits jobs to backend services via HTTP POST
- Polls backend service status endpoints at 2-second intervals
- Retrieves results from backend services when jobs complete
- Aggregates results from multiple services into unified response
- Manages job state in Redis with 1-hour TTL
- Handles partial failures gracefully

**Service Discovery:**

- scenes-service: `http://scenes-service:5002`
- faces-service: `http://faces-service:5003`
- semantics-service: `http://semantics-service:5004`
- objects-service: `http://objects-service:5005`

**Health Monitoring:**
All downstream services are health-checked via `<service>/health` endpoints with 5-second timeouts. The Vision API aggregates health status from all services.

### Processing Modes

#### Sequential Mode (Default)

Processes modules one at a time in dependency order:

**Execution Order:**

1. **Scenes Service** (if enabled) - Detects scene boundaries
2. **Faces Service** (if enabled) - Uses scene boundaries for optimized sampling
3. **Semantics Service** (if enabled) - Uses scene boundaries for frame selection
4. **Objects Service** (if enabled) - Independent processing

**Advantages:**

- Lower memory usage (one service active at a time)
- Predictable GPU resource usage (no contention)
- Clear error isolation
- Scene-aware face sampling (passes boundaries to faces-service)

**Performance:**

- 13.97 seconds for 120s video (scenes + faces)
- 3.39s scenes + 7.05s faces + 3.53s coordination overhead
- ~11.4 FPS aggregate throughput (CPU mode)

#### Parallel Mode (Future - Phase 2)

Processes all enabled modules simultaneously using `asyncio.gather()`:

**Advantages:**

- 3-4x faster completion time
- Efficient use of idle GPU during I/O
- Maximum throughput

**Considerations:**

- Higher GPU memory (all models loaded)
- Higher system memory usage
- Potential resource contention
- No inter-service dependency passing

### Module Configuration

Each module can be individually enabled with custom parameters:

**Scenes Module:**

```json
{
  "enable_scenes": true,
  "parameters": {
    "scene_detection_method": "content",
    "scene_threshold": 27.0,
    "min_scene_length": 0.6
  }
}
```

**Faces Module:**

```json
{
  "enable_faces": true,
  "parameters": {
    "face_min_confidence": 0.9,
    "max_faces": 50,
    "face_sampling_interval": 2.0,
    "enable_deduplication": true,
    "similarity_threshold": 0.6,
    "detect_demographics": true
  }
}
```

**Semantics Module (Phase 2):**

```json
{
  "enable_semantics": true
}
```

**Objects Module (Phase 3):**

```json
{
  "enable_objects": true
}
```

### Workflow Examples

#### Example 1: Scenes + Faces Sequential

```python
import requests
import time

# Submit job
response = requests.post("http://localhost:5010/vision/analyze", json={
    "source": "/media/videos/scene.mp4",
    "source_id": "12345",
    "enable_scenes": True,
    "enable_faces": True,
    "processing_mode": "sequential",
    "parameters": {
        "scene_threshold": 27.0,
        "face_min_confidence": 0.9,
        "max_faces": 50
    }
})

job_id = response.json()["job_id"]

# Poll status
while True:
    status = requests.get(f"http://localhost:5010/vision/jobs/{job_id}/status").json()
    print(f"Progress: {status['progress']*100:.1f}%")

    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["error"])

    time.sleep(2)

# Get results
results = requests.get(f"http://localhost:5010/vision/jobs/{job_id}/results").json()

# Access scene boundaries
scenes = results["scenes"]["scenes"]
for scene in scenes:
    print(f"Scene {scene['scene_number']}: {scene['start_timestamp']:.2f}s - {scene['end_timestamp']:.2f}s")

# Access detected faces
faces = results["faces"]["faces"]
for face in faces:
    print(f"Face {face['face_id']}: {len(face['detections'])} detections")
```

#### Example 2: Faces Only (No Scene Detection)

```python
response = requests.post("http://localhost:5010/vision/analyze", json={
    "source": "/media/videos/scene.mp4",
    "source_id": "12345",
    "enable_faces": True,
    "parameters": {
        "face_min_confidence": 0.8,
        "max_faces": 100,
        "face_sampling_interval": 1.0
    }
})
```

### Result Aggregation

The Vision API combines results from all enabled services into a single unified response:

```json
{
  "job_id": "16542ef6-73b5-4200-a5be-414b8bfb6bc2",
  "source_id": "test_rollup_001",
  "status": "completed",
  "scenes": {
    "status": "completed",
    "scenes": [
      {
        "scene_number": 0,
        "start_timestamp": 0.0,
        "end_timestamp": 30.03,
        "duration": 30.03
      }
    ],
    "metadata": {
      "processing_time_seconds": 3.39
    }
  },
  "faces": {
    "status": "completed",
    "faces": [
      {
        "face_id": "face_0",
        "embedding": [0.123, -0.456, ...],
        "detections": [...]
      }
    ],
    "metadata": {
      "unique_faces": 11,
      "total_detections": 16,
      "processing_time_seconds": 7.05
    }
  },
  "semantics": {
    "status": "not_implemented"
  },
  "objects": {
    "status": "not_implemented"
  },
  "metadata": {
    "processing_time_seconds": 13.97,
    "processing_mode": "sequential",
    "services_used": {
      "scenes": true,
      "faces": true,
      "semantics": false,
      "objects": false
    }
  }
}
```

### Job Listing

The Vision API provides a unified job listing endpoint that aggregates jobs from all services:

```python
import requests

# List all jobs
response = requests.get("http://localhost:5010/vision/jobs")
jobs = response.json()
print(f"Found {jobs['total']} jobs")

# Filter by status
completed = requests.get("http://localhost:5010/vision/jobs?status=completed")

# Filter by service
faces_jobs = requests.get("http://localhost:5010/vision/jobs?service=faces")

# Filter by source_id
scene_jobs = requests.get("http://localhost:5010/vision/jobs?source_id=12345")

# Date range filtering
from_date = requests.get(
    "http://localhost:5010/vision/jobs?start_date=2025-01-01T00:00:00Z"
)

# Include full results
with_results = requests.get(
    "http://localhost:5010/vision/jobs?include_results=true&limit=10"
)

# Pagination
page_2 = requests.get("http://localhost:5010/vision/jobs?limit=50&offset=50")
```

**Example Response:**

```json
{
  "jobs": [
    {
      "job_id": "16542ef6-73b5-4200-a5be-414b8bfb6bc2",
      "service": "vision",
      "status": "completed",
      "progress": 1.0,
      "source": "/media/videos/scene.mp4",
      "source_id": "12345",
      "created_at": "2025-11-26T04:30:00.000Z",
      "result_summary": {
        "scenes": 4,
        "faces": 11
      }
    },
    {
      "job_id": "faces-abc123",
      "service": "faces",
      "status": "processing",
      "progress": 0.75,
      "source": "/media/videos/another.mp4",
      "source_id": "67890",
      "created_at": "2025-11-26T04:35:00.000Z"
    }
  ],
  "total": 156,
  "limit": 50,
  "offset": 0
}
```

**Deduplication:**
Jobs are deduplicated by `cache_key`. If the same video is processed through both vision-api and directly to faces-service, only the vision-level job is returned (as it contains more complete context).

### Service Health Monitoring

The Vision API monitors all downstream services and reports aggregated health:

```python
# Health check all services
response = requests.get("http://localhost:5010/vision/health")

# Example response
{
  "status": "healthy",
  "service": "vision-api",
  "version": "1.0.0",
  "services": {
    "scenes": {
      "status": "healthy",
      "version": "1.0.0",
      "gpu_available": false
    },
    "faces": {
      "status": "healthy",
      "version": "1.0.0",
      "model": "buffalo_l",
      "gpu_available": false
    },
    "semantics": {
      "status": "healthy",
      "implemented": false,
      "phase": 2
    },
    "objects": {
      "status": "healthy",
      "implemented": false,
      "phase": 3
    }
  }
}
```

### Performance

**Test Results (CPU Mode - 120s Video):**

- Scenes detection: 3.39 seconds (4 scenes)
- Faces analysis: 7.05 seconds (11 frames)
- Total processing: 13.97 seconds
- Aggregate throughput: 11.4 FPS

**Expected GPU Performance:**

- Scenes detection: ~2s (GPU-accelerated histogram)
- Faces analysis: ~2-3s (GPU-accelerated InsightFace)
- Total processing: ~6-8 seconds (2-3x speedup)

### Configuration

Environment variables:

```bash
# Service URLs
SCENES_SERVICE_URL=http://scenes-service:5002
FACES_SERVICE_URL=http://faces-service:5003
SEMANTICS_SERVICE_URL=http://semantics-service:5004
OBJECTS_SERVICE_URL=http://objects-service:5005

# Redis
REDIS_URL=redis://redis:6379/0

# Cache Configuration
CACHE_TTL=31536000        # 1 year job history retention

# Timeouts
SERVICE_TIMEOUT=300        # 5 minutes per service call
POLLING_INTERVAL=2.0       # 2 seconds between status polls

# Logging
LOG_LEVEL=INFO
```

---

**Last Updated:** 2025-11-26
**Status:** Implemented and Tested
