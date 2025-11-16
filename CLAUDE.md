# Stash Auto Vision Service

**GPU-Accelerated Video Analysis Microservices for Stash**

A modular, high-performance video analysis platform providing face recognition, scene detection, semantic classification, and object detection for the Stash media organizer ecosystem.

---

## Project Overview

Stash Auto Vision is a standalone microservices platform that processes video content to extract faces, detect scene boundaries, and (future) provide semantic classification and object detection. It serves as the video processing backend for stash-compreface-plugin and future advanced scene analysis features.

### Core Principles

1. **Analysis Tool, Not Graphics Processor** - Services process video resources internally or accept metadata/paths, never transferring frame data over HTTP
2. **Batch-Oriented API** - Submit job → poll status → retrieve results (no per-frame HTTP calls)
3. **Decoupled Design** - Returns raw analysis data, consumers handle integration
4. **GPU-Accelerated** - CUDA-optimized for production, CPU-compatible for development
5. **Content-Based Caching** - Automatic invalidation on video changes, efficient reuse

### Key Capabilities

**Phase 1 (Implemented):**
- Video face detection and recognition (InsightFace buffalo_l, 99.86% accuracy)
- Production-grade face enhancement (CodeFormer, Nero-equivalent quality)
- Scene boundary detection (GPU-accelerated PySceneDetect)
- Frame extraction with multiple methods (OpenCV CUDA, PyAV, FFmpeg fallback)
- 512-D ArcFace embeddings with quality scoring
- Face clustering via embedding similarity (cosine distance)
- Content-based caching with SHA-256 keys
- Asynchronous job processing with progress tracking

**Future Phases:**
- **Phase 2:** CLIP-based scene understanding and semantic tagging
- **Phase 3:** YOLO-World open-vocabulary object detection
- **Phase 4:** Multi-modal search and advanced tagging
- **Phase 5:** stash-compreface-plugin integration

---

## Architecture

### Microservices Design

```
┌────────────────────────────────────────────────────────────┐
│                  Stash Auto Vision                          │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │  vision-api  │ :5010   │   Main Orchestrator      │    │
│  │              │◄────────┤   - Composite analysis   │    │
│  └──────┬───────┘         │   - Sequential workflow  │    │
│         │                 │   - Health aggregation   │    │
│         │                 └──────────────────────────┘    │
│         │                                                  │
│         ├─► frame-server   :5001 (Frame extraction)       │
│         ├─► scenes-service :5002 (Scene boundaries)       │
│         ├─► faces-service  :5003 (Face recognition)       │
│         ├─► semantics-svc  :5004 (CLIP) [Phase 2]         │
│         └─► objects-svc    :5005 (YOLO-World) [Phase 3]   │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │    redis     │ :6379   │   - Job metadata         │    │
│  │              │◄────────┤   - Result cache         │    │
│  └──────────────┘         │   - Content-based keys   │    │
│                           └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**vision-api (Orchestrator)**
- Accept job requests via `/vision/analyze` endpoint
- Coordinate multi-service workflows (scenes → faces sequential flow)
- Aggregate results from multiple services
- Health check aggregation across all services

**frame-server (Internal Service)**
- Multi-method frame extraction with per-frame fallback
- Primary: OpenCV CUDA/CPU (fastest)
- Fallback: PyAV hw/sw (robust FFmpeg bindings)
- Last resort: FFmpeg CLI
- WebVTT sprite sheet parsing
- On-demand frame serving with polling support
- Frame storage with TTL-based cleanup (cron job every hour)

**scenes-service**
- PySceneDetect integration with GPU-accelerated OpenCV backend
- ContentDetector, ThresholdDetector, AdaptiveDetector algorithms
- Scene boundary detection and timestamps
- Pass boundaries to downstream services (e.g., faces-service)

**faces-service**
- InsightFace (buffalo_l model on GPU, buffalo_l on CPU for accuracy parity)
- RetinaFace detection + ArcFace 512-D embeddings
- Face clustering via cosine similarity (threshold 0.6)
- Quality scoring and pose estimation (front, left, right, rotated)
- Optional demographics detection (age, gender)

**semantics-service** (Stubbed - Phase 2)
- CLIP (ViT-B/32) scene classification
- Zero-shot tagging and semantic search
- Scene embedding generation for similarity matching

**objects-service** (Stubbed - Phase 3)
- YOLO-World open-vocabulary object detection
- Custom category support
- Object tracking across frames

**redis**
- Job metadata storage (job_id, status, progress)
- Content-based cache key mapping
- Result caching with configurable TTL (default 1 hour)
- Bidirectional lookup (by job_id or cache_key)

---

## Technology Stack

### ML Models
- **InsightFace** (buffalo_l) - Face recognition, 99.86% LFW accuracy
- **PySceneDetect** - Scene boundary detection, 300-800 FPS on GPU
- **CLIP** (ViT-B/32) - Vision-language model [Phase 2]
- **YOLO-World** - Open-vocabulary object detection [Phase 3]

### Infrastructure
- **FastAPI** - Async web framework with automatic OpenAPI docs
- **Redis** - Job queue and result cache
- **Docker Compose** - 8-service orchestration
- **NVIDIA CUDA** - GPU acceleration (12.3.2)
- **OpenCV** - CUDA-accelerated video processing
- **PyAV** - FFmpeg Python bindings for robust frame extraction

### Deployment
- **GPU Mode:** NVIDIA runtime, buffalo_l model, CUDA providers
- **CPU Mode:** Standard runtime, buffalo_l model (accuracy parity), CPU providers
- **Port Configuration:** vision-api on 5010 (5000 conflicts with macOS AirPlay)

---

## Implementation Status

### Phase 1: Complete ✅

**Implemented Services (6/6):**
- [x] redis - Cache and job queue
- [x] dependency-checker - Health orchestration
- [x] frame-server - GPU-accelerated frame extraction
- [x] scenes-service - PySceneDetect integration
- [x] faces-service - InsightFace recognition
- [x] vision-api - Main orchestrator

**Stubbed Services (2/2):**
- [x] semantics-service - Returns "not_implemented" status
- [x] objects-service - Returns "not_implemented" status

**Core Features:**
- [x] Content-based caching with SHA-256 keys
- [x] Frame TTL management (2 hours with cron cleanup)
- [x] Sequential processing (scenes → faces)
- [x] Health check aggregation
- [x] GPU/CPU mode parity
- [x] Docker Compose deployment
- [x] OpenAPI documentation at `/docs`

**Testing:**
- [x] Service health checks passing
- [x] Face recognition validated with test videos
- [x] Scene detection validated
- [x] Cache hit/miss working correctly
- [x] Sequential workflow validated
- [x] Test data generated from Charades dataset

### Phase 2-5: Planned 🔄

See [Future Work](#future-work) section below.

---

## Service References

### API Documentation
- **[OpenAPI Specification](openapi.yml)** - Complete API schemas for all 6 services
- Access live docs: `http://localhost:5010/docs` (after starting services)

### Service-Specific Documentation
- **[Frame Server](docs/FRAME_SERVER.md)** - Frame extraction methods, sprite parsing, on-demand serving
- **[Scenes Service](docs/SCENES_SERVICE.md)** - PySceneDetect algorithms, boundary detection
- **[Faces Service](docs/FACES_SERVICE.md)** - InsightFace integration, clustering, embeddings
- **[Semantics Service](docs/SEMANTICS_SERVICE.md)** - CLIP integration [Phase 2]
- **[Objects Service](docs/OBJECTS_SERVICE.md)** - YOLO-World integration [Phase 3]

### User Guides
- **[How to Use](docs/HOW_TO_USE.md)** - Quick start, API examples, troubleshooting

### Infrastructure Documentation
- **[Docker Architecture](docs/DOCKER_ARCHITECTURE.md)** - Service topology, health checks, dependencies
- **[API Specification](docs/API_SPECIFICATION.md)** - All endpoints, request/response schemas
- **[Service Specifications](docs/SERVICE_SPECIFICATIONS.md)** - Implementation details per service

---

## Future Work

### Phase 2: Semantic Analysis (CLIP Integration)
**Duration:** 2-3 days

**Deliverables:**
- CLIP model integration (ViT-B/32)
- Scene classification with predefined tags
- Zero-shot tagging with custom prompts
- Scene embedding generation for similarity search
- Integration with scenes-service for boundary-aware processing

**Use Cases:**
- Auto-tag scenes by content type (indoor, outdoor, intimate, action, etc.)
- Semantic search: "find scenes with two people talking in a kitchen"
- Similar scene finder based on embedding distance

### Phase 3: Object Detection (YOLO-World Integration)
**Duration:** 2-3 days

**Deliverables:**
- YOLO-World model loading (medium variant)
- Open-vocabulary detection with custom categories
- Bounding box extraction and confidence scoring
- Object tracking across frames
- Temporal aggregation (objects present in video)

**Use Cases:**
- Detect furniture, props, locations automatically
- Custom object categories (user-defined)
- Safety/content filtering based on objects
- Action recognition via object interactions

### Phase 4: Production Hardening
**Duration:** 2-3 days

**Deliverables:**
- Retry logic with exponential backoff
- Graceful degradation (partial results on service failure)
- Structured logging (JSON format)
- Performance metrics and Prometheus integration
- Comprehensive unit and integration tests
- Stress testing with 10+ concurrent jobs

### Phase 5: stash-compreface-plugin Integration
**Duration:** 1-2 days

**Deliverables:**
- Update plugin to call vision-api instead of direct dlib processing
- Submit face recognition jobs and poll status
- Process results and create Compreface subjects
- Update scene performers in Stash database
- End-to-end workflow validation
- Performance comparison vs dlib (target: 3-5x faster)

---

## Development Notes

### Critical Configuration

**Port Configuration:**
- vision-api: **5010** (not 5000 - conflicts with macOS AirPlay Receiver)
- Other services: 5001-5005 as documented

**Media Mounts:**
- Single media mount point: `/media/videos` (mapped to SERVER_MEDIA_PATH in .env)
- Test data location: `tests/data/` (Charades dataset, compound test videos)
- CMU samples: `tests/data/cmu/` (face recognition validation)

**Frame Storage:**
- Temporary storage: `/tmp/frames/` inside containers
- TTL: 2 hours (configurable via FRAME_TTL_HOURS)
- Cleanup: Cron job runs hourly inside frame-server

**Confidence Thresholds:**
- Environment variables set default confidence thresholds for each service
- Per-request parameters override environment defaults
- Precedence: Request parameter > Environment variable > Hard-coded default

| Service | Request Parameter | Environment Variable | Default | Range | Notes |
|---------|------------------|---------------------|---------|-------|-------|
| Faces | `face_min_confidence` | `FACES_MIN_CONFIDENCE` | 0.9 | 0.0-1.0 | Lower for challenging lighting (0.7-0.8) |
| Scenes | `scene_threshold` | `SCENES_THRESHOLD` | 27.0 | 0.0-100.0 | PySceneDetect ContentDetector scale |
| Semantics | `semantics_min_confidence` | `SEMANTICS_MIN_CONFIDENCE` | 0.5 | 0.0-1.0 | CLIP classification (Phase 2) |
| Objects | `objects_min_confidence` | `OBJECTS_MIN_CONFIDENCE` | 0.5 | 0.0-1.0 | YOLO detection (Phase 3) |

Example: Set `FACES_MIN_CONFIDENCE=0.7` in `.env` for lower quality videos, or override per-request with `face_min_confidence` parameter.

### Caching Strategy

**Content-Based Keys:**
```python
cache_key = SHA256(video_path + mtime + module + params)
```

**Features:**
- Automatic invalidation when video file changes (mtime tracking)
- Bidirectional lookup by job_id or cache_key
- Module-specific namespacing prevents collisions
- Configurable TTL (default 3600 seconds = 1 hour)

**Redis Structure:**
```
{module}:job:{job_id}:status       # Job metadata
{module}:job:{job_id}:result       # Job results
{module}:cache:{cache_key}:job_id  # Cache key → job_id mapping
```

See [docs/CACHE_STRATEGY.md](docs/CACHE_STRATEGY.md) for complete details.

### Sequential Processing

**Default Mode:** Sequential (not parallel)
- Avoids GPU memory contention
- Predictable resource usage
- Services pass data to next stage (e.g., scene boundaries from scenes → faces)

**Processing Flow:**
```
1. vision-api receives job
2. scenes-service detects boundaries
3. faces-service receives boundaries, processes only relevant frames
4. vision-api aggregates results
```

**Advantages:**
- Lower peak memory usage
- GPU can focus on one model at a time
- Debugging easier (clear execution order)
- Scene boundaries optimize face detection (skip static scenes)

### Testing

**Test Data Location:** `tests/data/`
- **Charades:** ~9,500 videos for frame/scene testing
- **Compound Videos:** Generated test videos with specific characteristics
- **CMU Multi-PIE:** Face recognition validation dataset

**Quick Testing:**
```bash
# Start services
docker-compose up -d

# Health check
curl http://localhost:5010/health

# Run face detection on test video
curl -X POST http://localhost:5010/vision/analyze/faces \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/compound/faces-service/single_person_varied_conditions.mp4",
    "scene_id": "test_001"
  }'
```

See [docs/TESTING.md](docs/TESTING.md) for comprehensive test scenarios.

### Deployment Modes

**Development (CPU Mode):**
```bash
cp .env.cpu.example .env
docker-compose up -d
```
- Uses Dockerfile.cpu variants
- buffalo_l model (same as GPU for accuracy parity)
- OpenCV CPU backend
- Port 5010 for vision-api (macOS compatibility)

**Production (GPU Mode):**
```bash
cp .env.example .env
# Edit .env with production paths
docker-compose up -d
```
- Uses Dockerfile (GPU) variants
- NVIDIA runtime required
- buffalo_l model with CUDA providers
- OpenCV CUDA backend
- Port 5000 for vision-api

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete setup instructions.

---

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd stash-auto-vision

# Create Docker network
docker network create stash-auto-vision-network

# Configure environment (choose one)
cp .env.cpu.example .env    # For development
cp .env.example .env        # For production (edit paths)

# Start services
docker-compose up -d

# Verify health
curl http://localhost:5010/health

# View logs
docker-compose logs -f vision-api

# Access OpenAPI docs
open http://localhost:5010/docs
```

---

## Performance Benchmarks

### GPU Mode (RTX A4000)

| Operation | Target | Actual (Test Results) |
|-----------|--------|----------------------|
| Frame extraction | 200-400 FPS | N/A (CPU testing only) |
| Scene detection | 300-800 FPS | N/A (CPU testing only) |
| Face detection | ~30 FPS | N/A (CPU testing only) |
| Face embedding | <10ms/face | N/A (CPU testing only) |

### CPU Mode (Mac Development)

| Operation | Target | Actual (Test Results) |
|-----------|--------|----------------------|
| Frame extraction | 30-60 FPS | ~14 FPS (60s video, 2s interval, 30 frames in 2.12s) |
| Scene detection | 100-500 FPS | ~150 FPS (90s video, 3.9s processing) |
| Face detection | ~5 FPS | ~7 FPS (60s video, 8.5s processing) |

Note: CPU results from macOS development environment. GPU performance to be validated on production hardware.

---

## Troubleshooting

**Services not starting:**
```bash
docker-compose logs dependency-checker
docker-compose logs <service-name>
```

**GPU not detected (production):**
```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
docker-compose logs faces-service | grep -i cuda
```

**Cache not working:**
```bash
docker exec -it vision-redis redis-cli KEYS "*"
```

**Port conflicts (macOS):**
- Default port 5000 conflicts with AirPlay Receiver
- Solution: Use VISION_API_PORT=5010 (already configured in .env.cpu.example)

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for comprehensive troubleshooting.

---

## Contributing

(TBD - contribution guidelines)

---

## License

(TBD - license information)

---

**Status:** Phase 1 Complete - Ready for Production Testing
**Version:** 1.0.0
**Last Updated:** 2025-11-09
