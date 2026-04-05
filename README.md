# Stash Auto Vision

**GPU-Accelerated Video Analysis Microservices for Stash**

A modular, high-performance video analysis platform providing face recognition, scene detection, semantic classification, and object detection for the Stash media organizer ecosystem.

---

## Features

### Phase 1-3 (Complete)

- ✅ **Face Recognition** - InsightFace (99.86% accuracy, 512-D embeddings)
- ✅ **Face Enhancement** - Optional CodeFormer/GFPGAN for low-quality faces
- ✅ **Quality-Based Filtering** - Three-tier quality system with enhancement triggering
- ✅ **Scene Detection** - GPU-accelerated PySceneDetect (300-800 FPS)
- ✅ **Frame Server** - Multi-method extraction with PyAV fallback
- ✅ **Smart Caching** - Content-based Redis caching with automatic invalidation
- ✅ **Tag Classification** - Trained bi-encoder classifier (99.2% match rate)
- ✅ **Jobs Viewer** - React-based UI for monitoring and browsing job results

### Phase 4+ (Planned)

- 🔄 **Object Detection** - YOLO-World open-vocabulary detection
- 🔄 **Production Hardening** - Retry logic, metrics, comprehensive testing

---

## Quick Start

### Prerequisites

**Production (Unraid):**

- NVIDIA GPU (RTX 3060+ with 12GB+ VRAM)
- CUDA 12.x drivers
- Docker with NVIDIA runtime

**Development (Mac):**

- Docker Desktop 4.0+
- 16GB+ RAM

### Installation

```bash
# Clone repository
git clone <repo-url>
cd stash-auto-vision

# Create Docker network
docker network create stash-auto-vision-network

# GPU Mode (Production)
cp .env.example .env
# Edit .env: Set SERVER_MEDIA_PATH=/mnt/user/movies (or your path)
docker-compose up -d

# CPU Mode (Development)
cp .env.cpu.example .env
# Edit .env: Paths already configured for tests/data
docker-compose up -d

# Verify all services healthy
curl http://localhost:5010/health | jq .

# Access OpenAPI documentation
open http://localhost:5010/docs
```

### Quick Test

```bash
# Test face recognition on sample video
curl -X POST http://localhost:5003/faces/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/compound/faces-service/single_person_varied_conditions.mp4",
    "source_id": "quick_test_001",
    "parameters": {
      "min_confidence": 0.9,
      "enable_deduplication": true
    }
  }' | jq .

# Poll for results
JOB_ID="<paste-job-id-from-response>"
curl "http://localhost:5003/faces/jobs/$JOB_ID/status" | jq .
curl "http://localhost:5003/faces/jobs/$JOB_ID/results" | jq .
```

---

## Architecture

```
┌──────────────┐
│  vision-api  │ :5010 ← Main API (orchestrator)
└──────┬───────┘
       │
       ├─► frame-server   :5001 (internal frame extraction)
       ├─► scenes-service :5002 (scene boundary detection)
       ├─► faces-service  :5003 (face recognition)
       ├─► semantics-svc  :5004 (Tag classifier) ✅
       └─► objects-svc    :5005 (YOLO detection) [Phase 4]
```

**10 Microservices:**

- **redis** - Cache and job queue
- **dependency-checker** - Health orchestration
- **frame-server** - GPU-accelerated frame extraction
- **scenes-service** - Scene boundary detection
- **faces-service** - Face recognition
- **semantics-service** - Tag classifier (captioning + classification) ✅
- **objects-service** - Object detection (stubbed)
- **vision-api** - Main orchestrator
- **schema-service** - OpenAPI aggregation
- **jobs-viewer** - React monitoring UI

---

## API Examples

### Analyze Faces

```bash
# Submit job
curl -X POST http://localhost:5003/faces/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "parameters": {
      "min_confidence": 0.9,
      "max_faces": 50
    }
  }'

# Response
{
  "job_id": "faces-550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2025-11-08T12:34:56.789Z"
}
```

### Check Status

```bash
curl http://localhost:5003/faces/jobs/{job_id}/status
```

### Get Results

```bash
curl http://localhost:5003/faces/jobs/{job_id}/results
```

### Comprehensive Analysis

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "modules": {
      "scenes": {"enabled": true},
      "faces": {"enabled": true}
    }
  }'
```

---

## Documentation

- **[How to Use](docs/HOW_TO_USE.md)** - Quick start, API examples, troubleshooting
- **[Project Specification](CLAUDE.md)** - Architecture, status, development notes
- **[OpenAPI Specification](http://localhost:5009/openapi.yaml)** - Complete API schemas for all services
- **[Frame Server](docs/FRAME_SERVER.md)** - Frame extraction, sprite parsing, on-demand serving
- **[Scenes Service](docs/SCENES_SERVICE.md)** - PySceneDetect algorithms, boundary detection
- **[Testing Guide](docs/TESTING.md)** - Test scenarios and results

---

## Performance

### GPU Mode (RTX A4000)

| Operation          | Performance    |
| ------------------ | -------------- |
| Frame extraction   | ~200-400 FPS   |
| Scene detection    | 300-800 FPS    |
| Face detection     | ~30 FPS        |
| Face embedding     | <10ms per face |
| Full video (5 min) | ~5 minutes     |

### CPU Mode (Mac Development)

| Operation          | Performance |
| ------------------ | ----------- |
| Frame extraction   | ~30-60 FPS  |
| Scene detection    | 100-500 FPS |
| Face detection     | ~5 FPS      |
| Full video (5 min) | ~40 minutes |

---

## Technology Stack

**ML Models:**

- InsightFace (buffalo_l) - Face recognition
- PySceneDetect - Scene boundaries
- Tag Classifier (bi-encoder) - Tag classification ✅
- JoyCaption (beta-one) - Frame captioning ✅
- YOLO-World - Object detection [Phase 4]

**Infrastructure:**

- FastAPI - Web framework
- Redis - Caching & job queue
- Docker - Containerization
- CUDA - GPU acceleration

---

## Configuration

### GPU Mode (.env)

```bash
SERVER_MEDIA_PATH=/mnt/user/movies
DOCKER_RUNTIME=nvidia
INSIGHTFACE_DEVICE=cuda
OPENCV_DEVICE=cuda
```

### CPU Mode (.env)

```bash
SERVER_MEDIA_PATH=./tests/data
DOCKER_RUNTIME=runc
INSIGHTFACE_DEVICE=cpu
OPENCV_DEVICE=cpu
VISION_API_PORT=5010  # macOS: port 5000 conflicts with AirPlay
```

### Confidence Thresholds

Environment variables set default confidence thresholds, overridable per-request:

```bash
FACES_MIN_CONFIDENCE=0.9      # face_min_confidence parameter (0.7-0.8 for low quality)
SCENES_THRESHOLD=27.0          # scene_threshold parameter (PySceneDetect scale)
SEMANTICS_MIN_CONFIDENCE=0.75  # semantics_min_confidence parameter (Phase 3)
OBJECTS_MIN_CONFIDENCE=0.5     # objects_min_confidence parameter (Phase 4)
```

Request parameter names are service-prefixed to avoid collisions in the vision-api rollup endpoint.

See [Deployment Guide](docs/DEPLOYMENT.md) for full configuration options.

---

## Development

```bash
# Start in dev mode (hot reload)
docker-compose up -d

# View logs
docker-compose logs -f vision-api

# Rebuild after changes
docker-compose up -d --build faces-service

# Run tests
docker-compose exec vision-api pytest

# Access Redis CLI
docker exec -it vision-redis redis-cli
```

---

## Integration

### With stash-compreface-plugin

Stash Auto Vision serves as the video processing backend for the stash-compreface-plugin:

1. Plugin submits video analysis job to vision-api
2. Vision service processes video (face detection, scene analysis)
3. Plugin receives results (faces, embeddings, timestamps)
4. Plugin sends faces to Compreface for matching
5. Plugin updates Stash performers

See [stash-compreface-plugin](../stash-compreface-plugin/CLAUDE.md) for integration details.

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Check service logs
docker-compose logs faces-service | grep -i cuda
```

### Services Not Starting

```bash
# Check dependency-checker health
docker compose logs dependency-checker

# Manually check service health
curl http://localhost:5001/health
curl http://localhost:5010/health
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Use sequential processing
echo "PROCESSING_MODE=sequential" >> .env
docker-compose up -d --force-recreate
```

See [Deployment Guide](docs/DEPLOYMENT.md) for comprehensive troubleshooting.

---

## Roadmap

### Phase 1: Core Services ✅ (Complete)

- [x] Planning documentation
- [x] Frame server implementation
- [x] Scenes service implementation
- [x] Faces service implementation
- [x] Vision API orchestrator
- [x] End-to-end testing
- [x] Production-ready deployment

### Phase 2 & 3: Tag Classifier ✅ (Complete)

- [x] Frame captioning (JoyCaption beta-one)
- [x] LLM narrative summary (Llama 3.1 8B)
- [x] Trained bi-encoder tag classification (99.2% match rate)
- [x] Taxonomy pre-loading from Stash

### Phase 4: Object Detection 🔄

- [ ] YOLO-World integration
- [ ] Open-vocabulary detection
- [ ] Custom object categories
- [ ] Object tracking

### Phase 5: Production Hardening 🔄

- [ ] Error handling & retry logic
- [ ] Monitoring & metrics
- [ ] Performance optimization
- [ ] Comprehensive testing

### Phase 6: Plugin Integration 🔄

- [ ] stash-compreface-plugin update
- [ ] Real-world validation
- [ ] Performance benchmarking

---

## Contributing

(TBD - contribution guidelines)

---

## License

(TBD - license information)

---

## Credits

**Technology:**

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Scene detection
- [JoyCaption](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) - Frame captioning VLM
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) - Object detection
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

**Ecosystem:**

- [Stash](https://github.com/stashapp/stash) - Media organizer
- [Compreface](https://github.com/exadel-inc/CompreFace) - Face recognition service

---

**Status:** Phase 1-3 Complete - Tag Classifier Pipeline
**Version:** 3.0.0
**Last Updated:** 2026-04-04
