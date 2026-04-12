# Stash Auto Vision Service

**GPU-Accelerated Video Analysis Microservices for Stash**

A modular, high-performance video analysis platform providing face recognition, scene detection, semantic classification, and object detection for the Stash media organizer ecosystem.

---

## Project Overview

Stash Auto Vision is a standalone microservices platform that processes video content to extract faces, detect scene boundaries, provide semantic classification, and (future) object detection. It serves as the video processing backend for two Stash plugins: `stash-auto-vision-tagging` (fully integrated, tag classification via semantics-service) and `stash-compreface-plugin` (partially integrated, face recognition via faces-service — pending final end-to-end testing).

### Core Principles

1. **Analysis Tool, Not Graphics Processor** - Services process video resources internally or accept metadata/paths, never transferring frame data over HTTP
2. **Batch-Oriented API** - Submit job → poll status → retrieve results (no per-frame HTTP calls)
3. **Decoupled Design** - Returns raw analysis data, consumers handle integration
4. **GPU-Accelerated** - CUDA-optimized for production, CPU-compatible for development
5. **Content-Based Caching** - Automatic invalidation on video changes, efficient reuse

### Key Capabilities

**Phase 1 (Complete ✅):**

- Video face detection and recognition (InsightFace buffalo_l, 99.86% accuracy)
- Optional face enhancement via CodeFormer/GFPGAN (production-grade quality)
- Three-tier quality system (detection confidence, quality trigger, minimum quality)
- Scene boundary detection (GPU-accelerated PySceneDetect)
- Frame extraction with multiple methods (OpenCV CUDA, PyAV, FFmpeg fallback)
- 512-D ArcFace embeddings with quality scoring
- Face clustering via embedding similarity (cosine distance)
- Content-based caching with SHA-256 keys
- Asynchronous job processing with progress tracking

**Phase 2 & 3 (Complete ✅):**

- Trained multi-view bi-encoder tag classifier (99.2% match rate)
- Pipeline: scene resolution from Stash, sprite frame extraction, JoyCaption beta-one captioning, Llama 3.1 8B narrative summary, tag classification
- Taxonomy pre-loading from Stash at startup via STASH_URL + SEMANTICS_TAG_ID
- Scene data fetched from Stash via findScene(source_id) — sprites, details, performers
- Sprite sheets as default frame source; scene-based/interval as fallback
- Resource manager for GPU orchestration
- Replaces previous SigLIP zero-shot and JoyCaption alpha-two captioning services

**Future Phases:**

- **Phase 4:** YOLO-World open-vocabulary object detection **and** Flash Attention 2 integration for the semantics runtimes (LlamaRuntime + JoyCaption) — lossless VRAM savings and 1.5–2× speedup
- **Phase 5:** Production hardening (retries, metrics, stress tests)
- **Phase 6:** Finalize `stash-compreface-plugin` video-path integration (end-to-end testing)

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
│         ├─► semantics-svc  :5004 (Tag Classifier) ✅      │
│         └─► objects-svc    :5005 (YOLO-World) [Phase 4]   │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │resource-mgr  │ :5007   │   - GPU orchestration    │    │
│  │              │◄────────┤   - VRAM reservation     │    │
│  └──────────────┘         │   - Lease management     │    │
│                           └──────────────────────────┘    │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │    redis     │ :6379   │   - Job metadata         │    │
│  │              │◄────────┤   - Result cache         │    │
│  └──────────────┘         │   - Content-based keys   │    │
│                           └──────────────────────────┘    │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │ schema-svc   │ :5009   │   - OpenAPI aggregation  │    │
│  │              │◄────────┤   - Combined Swagger UI  │    │
│  └──────────────┘         └──────────────────────────┘    │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐    │
│  │ jobs-viewer  │ :5020   │   - React monitoring UI  │    │
│  │              │◄────────┤   - Job browsing/status  │    │
│  └──────────────┘         └──────────────────────────┘    │
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

- InsightFace buffalo_l with multi-size detection (320/640/1024 det_size auto-selection)
- RetinaFace detection + ArcFace 512-D embeddings
- Optional face enhancement (CodeFormer/GFPGAN) for low-quality detections
- Three-tier quality gate: detection confidence, quality trigger, minimum quality
- Face clustering via cosine similarity (threshold 0.6)
- Quality scoring: Laplacian variance sharpness + size/pose/occlusion components
- Sprite sheet integration for ultra-fast processing
- Optional demographics detection (age, gender)

**semantics-service** (Phase 3 Complete ✅)

- Trained multi-view bi-encoder tag classifier (99.2% match rate)
- Scene data resolved from Stash via `source_id` (sprites, details, performers)
- Sprite sheets as default frame source (no video decoding required)
- JoyCaption beta-one per-frame captioning (~8GB VRAM, loaded/unloaded per job)
- Llama 3.1 8B narrative summary via external API (with scene metadata)
- Tag classifier ~1.4GB VRAM (kept loaded)
- Taxonomy pre-loaded from Stash at startup
- Replaces previous SigLIP and JoyCaption alpha-two services

**objects-service** (Stubbed - Phase 4)

- YOLO-World open-vocabulary object detection
- Custom category support
- Object tracking across frames

**resource-manager**

- GPU/VRAM orchestration across all services
- Lease-based allocation with priority queue
- Heartbeat-based timeout for abandoned resources
- Fair scheduling (FIFO within priority levels)
- Automatic detection of GPU hardware info

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
- **Tag Classifier** - Trained multi-view bi-encoder (99.2% match rate) ✅
- **JoyCaption** (beta-one) - Frame captioning VLM ✅
- **YOLO-World** - Open-vocabulary object detection [Phase 4]

### Infrastructure

- **FastAPI** - Async web framework with automatic OpenAPI docs
- **Redis** - Job queue and result cache
- **Docker Compose** - 10-service orchestration
- **NVIDIA CUDA** - GPU acceleration (12.3.2)
- **OpenCV** - CUDA-accelerated video processing
- **PyAV** - FFmpeg Python bindings for robust frame extraction

### Deployment

- **GPU Mode:** NVIDIA runtime, buffalo_l model, CUDA providers
- **CPU Mode:** Standard runtime, buffalo_l model (accuracy parity), CPU providers
- **Port Configuration:** vision-api on 5010 (5000 conflicts with macOS AirPlay)

---

## Implementation Status

### Phase 1, 2 & 3: Complete ✅

**Implemented Services (10/11):**

- [x] redis - Cache and job queue
- [x] dependency-checker - Health orchestration
- [x] frame-server - GPU-accelerated frame extraction
- [x] scenes-service - PySceneDetect integration
- [x] faces-service - InsightFace recognition
- [x] semantics-service - Tag classifier pipeline (captioning merged in)
- [x] resource-manager - GPU orchestration
- [x] vision-api - Main orchestrator
- [x] schema-service - OpenAPI aggregation
- [x] jobs-viewer - React monitoring UI

**Stubbed Services (1/11):**

- [x] objects-service - Returns "not_implemented" status (Phase 4)

**Core Features:**

- [x] Content-based caching with SHA-256 keys
- [x] Frame TTL management (2 hours with cron cleanup)
- [x] Sequential processing (scenes → faces)
- [x] Face enhancement with CodeFormer/GFPGAN (optional)
- [x] Three-tier quality system with enhanced flag tracking
- [x] Sprite sheet integration with cross-service volume sharing
- [x] InsightFace model persistence (startup time: 15s → 3s)
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

### Phase 4-6: Planned 🔄

See [Future Work](#future-work) section below.

---

## Service References

### API Documentation

- **OpenAPI Specification** - Auto-generated from FastAPI at runtime (`/openapi.json`)
- Access combined live docs: `http://localhost:5009/docs` (Swagger UI aggregated by schema-service)
- Per-service schemas: `http://localhost:500{1..5}/openapi.json`
- **Jobs Viewer** - React-based job monitoring UI at `http://localhost:5020`

### Service-Specific Documentation

- **[Frame Server](docs/FRAME_SERVER.md)** - Frame extraction methods, sprite parsing, on-demand serving
- **[Scenes Service](docs/SCENES_SERVICE.md)** - PySceneDetect algorithms, boundary detection
- **[Faces Service](docs/FACES_SERVICE.md)** - InsightFace integration, clustering, embeddings
- **[Semantics Service](docs/SEMANTICS_SERVICE.md)** - Tag classifier pipeline (Phase 3)
- **[Resource Manager](docs/RESOURCE_MANAGER.md)** - GPU orchestration
- **[Objects Service](docs/OBJECTS_SERVICE.md)** - YOLO-World integration (Phase 4)

### User Guides

- **[How to Use](docs/HOW_TO_USE.md)** - Quick start, API examples, troubleshooting
- **[Vision API](docs/VISION_API.md)** - Orchestrator request shape, aggregated results, job listing
- **[Testing Guide](docs/TESTING.md)** - Test scenarios and results

> **Note:** A previous generation of this project had several unified infrastructure docs (`DOCKER_ARCHITECTURE.md`, `API_SPECIFICATION.md`, `SERVICE_SPECIFICATIONS.md`, `CACHE_STRATEGY.md`, `DEPLOYMENT.md`). Those documents were dissolved into the per-service markdowns listed above, plus `docker-compose.yml` and `.env` examples in the repo root. The live OpenAPI schema at `http://localhost:5009/docs` is the authoritative API reference.

---

## Completed Work

### Phase 2 & 3: Semantic Analysis (Tag Classifier) ✅

**Status:** COMPLETE (2026-04-04)

> **Phase history note:** Phase 2 and Phase 3 are collapsed into one body of work because Phase 2's original design (SigLIP zero-shot classification plus a JoyCaption alpha-two captioning service) was abandoned partway through and redesigned as the Phase 3 trained multi-view bi-encoder classifier with JoyCaption beta-one captioning. References to "Phase 2 & 3" throughout these docs reflect that combined implementation effort, not two separate milestones.

**Implemented:**

- Trained multi-view bi-encoder tag classifier (99.2% match rate)
- Pipeline: frame extraction, JoyCaption beta-one captioning, Llama 3.1 8B narrative summary, tag classification
- Taxonomy pre-loading from Stash at startup via STASH_URL + SEMANTICS_TAG_ID
- JoyCaption beta-one VLM (~8GB VRAM, loaded/unloaded per job)
- Llama 3.1 8B narrative summary via external API
- Tag classifier ~1.4GB VRAM (kept loaded)
- Integration with scenes-service for boundary-aware processing
  - Vision-API orchestration (primary method)
  - Standalone scenes_job_id parameter (optional method)
- Content-based caching via Redis

**Achieved Use Cases:**

- Auto-tag scenes against Stash taxonomy with high accuracy (99.2% match rate)
- Per-scene narrative summaries from frame captions
- Scene-aware frame sampling and aggregation

---

## Future Work

### Phase 4: Object Detection (YOLO-World Integration)

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

**Additional Phase 4 deliverable: Flash Attention 2 for the semantics runtimes**

Enable the `flash_attention_2` kernel on both the shared `LlamaRuntime` (used for scene summary + suggested-title generation) and the JoyCaption `CaptionGenerator`. Flash-attention is a mathematically equivalent attention kernel, so there is no impact on model outputs or classifier accuracy — the win is entirely in memory layout and compute efficiency. This is the only known lossless VRAM optimization that does not require swapping models or retraining the classifier, so it is gated on the existing quality envelope and can ship without any of the model-swap caveats in "Future Work → Why not a smaller Llama?".

**Deliverables:**

- Pass `attn_implementation="flash_attention_2"` to `AutoModelForCausalLM.from_pretrained()` in `semantics-service/app/llama_runtime.py` (CUDA-only; fall back to PyTorch SDPA on CPU automatically via a conditional)
- Pass the same flag to the JoyCaption load in `semantics-service/app/caption_generator.py`
- Add the `flash-attn` wheel to the semantics-service GPU Dockerfile, pinned to the container's CUDA + PyTorch versions at build time
- Re-measure and publish peak VRAM numbers in `docs/SEMANTICS_SERVICE.md` — expected: semantics-alone peak drops from ~11.4 GB (classifier + Llama 8-bit) to ~10 GB
- Publish a "VRAM by workload" tier matrix in `README.md` and `docs/SEMANTICS_SERVICE.md` so constrained-GPU users can see what each card tier can comfortably run, rather than reading a single conservative 16 GB floor
- Verify equivalence: run a small regression suite comparing pre- and post-flash-attention scene summaries + suggested titles + classifier outputs on a fixed test set; outputs should be bit-close (differences only at the level of floating-point accumulation order)

**Benefits:**

- **Lossless** — no classifier retrain, no quality regression, no change to JoyCaption caption style that would shift the classifier's input distribution
- ~500 MB – 1.5 GB VRAM savings on the KV-cache during 1024-token summary generation on Llama 3.1 8B
- 1.5–2× speedup on Llama summary generation and JoyCaption frame captioning
- Lowers the effective semantics-alone VRAM floor into the 10–11 GB range, making 12 GB consumer GPUs (RTX 3080 Ti, 4070, 4070 Super) comfortable for semantics-only workloads
- No-op on CPU mode — CPU runtime falls back to SDPA without errors

**Use Cases:**

- Users on 12 GB consumer GPUs who currently can't comfortably run the semantics pipeline
- Faster batch processing on existing 16 GB deployments
- More headroom for concurrent GPU usage (e.g. faces + semantics serialized through the resource-manager)
- Unblocks a "real" VRAM tier documentation story (10 / 12 / 16 GB tiers) instead of a single scary 16 GB number that's driving user complaints

### Phase 5: Production Hardening

**Deliverables:**

- Retry logic with exponential backoff
- Graceful degradation (partial results on service failure)
- Structured logging (JSON format)
- Performance metrics and Prometheus integration
- Comprehensive unit and integration tests
- Stress testing with 10+ concurrent jobs

### Phase 6: Finalize Compreface Plugin Integration

The `stash-auto-vision-tagging` plugin (see `../stash-auto-vision-tagging/`) is **fully integrated** — tag classification jobs submit against either `/vision/analyze` or `/semantics/analyze` with automatic submit-time failover, and results are written back to Stash via GraphQL. That work is considered complete.

The `stash-compreface-plugin` (see `../stash-compreface-plugin/`) is **partially integrated** — its video recognition path (scene face extraction, sprite processing, embedding-based matching, face enhancement) is wired to this project's faces-service / frame-server as its Vision Service backend, but has not yet been validated end-to-end.

**Deliverables:**

- End-to-end validation of the compreface plugin's video recognition path against a live Stash instance
- Benchmark performance vs. the plugin's legacy dlib path
- Document remaining rough edges discovered during testing

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
- Sprite storage: `/tmp/sprites/` (shared volume for sprite tiles)
- TTL: 2 hours (configurable via FRAME_TTL_HOURS)
- Cleanup: Cron job runs hourly inside frame-server

**Model Caching:**

- InsightFace models: `/root/.insightface` (~275MB, persistent volume)
- Startup time: ~15s initial download, ~3s with cache

**Confidence Thresholds:**

- Environment variables set default confidence thresholds for each service
- Per-request parameters override environment defaults
- Precedence: Request parameter > Environment variable > Hard-coded default

| Service   | Request Parameter           | Environment Variable                | Default | Range     | Notes                                        |
| --------- | --------------------------- | ----------------------------------- | ------- | --------- | -------------------------------------------- |
| Faces     | `face_min_confidence`       | `FACES_MIN_CONFIDENCE`              | 0.9     | 0.0-1.0   | Detection threshold (0.7 CPU, 0.9 GPU)       |
| Faces     | `face_min_quality`          | `FACES_MIN_QUALITY`                 | 0.0     | 0.0-1.0   | Minimum quality to keep (0.0 = no filtering) |
| Faces     | N/A (nested in enhancement) | `FACES_ENHANCEMENT_QUALITY_TRIGGER` | 0.5     | 0.0-1.0   | Trigger enhancement if quality below this    |
| Scenes    | `scene_threshold`           | `SCENES_THRESHOLD`                  | 27.0    | 0.0-100.0 | PySceneDetect ContentDetector scale          |
| Semantics | `semantics_min_confidence`  | `SEMANTICS_MIN_CONFIDENCE`          | 0.75    | 0.0-1.0   | Tag classifier (Phase 3 complete)            |
| Objects   | `objects_min_confidence`    | `OBJECTS_MIN_CONFIDENCE`            | 0.5     | 0.0-1.0   | YOLO detection (Phase 4, stub)               |

Example: Set `FACES_MIN_CONFIDENCE=0.7` in `.env` for lower quality videos, or override per-request with `face_min_confidence` parameter.

**Face Enhancement Parameters:**

- `enhancement.enabled`: Enable face enhancement (default: false)
- `enhancement.quality_trigger`: Trigger enhancement if quality below this threshold (default: 0.5)
- `enhancement.model`: "codeformer" (recommended) or "gfpgan" (default: "codeformer")
- `enhancement.fidelity_weight`: Quality vs fidelity tradeoff, 0.0-1.0 (default: 0.5)

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

Cache behavior is implemented by each service's `cache_manager.py` (see `vision-api/app/cache_manager.py`, `faces-service/app/cache_manager.py`, etc.) — there is no centralized cache spec document.

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
curl http://localhost:5010/vision/health

# Run face detection on test video
curl -X POST http://localhost:5010/vision/analyze/faces \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/compound/faces-service/single_person_varied_conditions.mp4",
    "source_id": "test_001"
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

See `docker-compose.yml`, `.env.example`, and `.env.cpu.example` in the repo root for complete setup.

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
curl http://localhost:5010/vision/health

# View logs
docker-compose logs -f vision-api

# Access combined OpenAPI docs (aggregated by schema-service)
open http://localhost:5009/docs
```

---

## Performance Benchmarks

### GPU Mode (RTX A4000)

| Operation        | Target      | Actual (Test Results)  |
| ---------------- | ----------- | ---------------------- |
| Frame extraction | 200-400 FPS | N/A (CPU testing only) |
| Scene detection  | 300-800 FPS | N/A (CPU testing only) |
| Face detection   | ~30 FPS     | N/A (CPU testing only) |
| Face embedding   | <10ms/face  | N/A (CPU testing only) |

### CPU Mode (Mac Development)

| Operation        | Target      | Actual (Test Results)                                |
| ---------------- | ----------- | ---------------------------------------------------- |
| Frame extraction | 30-60 FPS   | ~14 FPS (60s video, 2s interval, 30 frames in 2.12s) |
| Scene detection  | 100-500 FPS | ~150 FPS (90s video, 3.9s processing)                |
| Face detection   | ~5 FPS      | ~7 FPS (60s video, 8.5s processing)                  |

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

See the service-specific docs in `docs/` for further troubleshooting.

---

## Contributing

(TBD - contribution guidelines)

---

## License

(TBD - license information)

---

**Status:** Phase 1-3 Complete - Tag Classifier Pipeline
**Version:** 3.0.0
**Last Updated:** 2026-04-04

> **API Documentation:** OpenAPI schemas are auto-generated from FastAPI at runtime (`/openapi.json`). The schema-service at port 5009 aggregates all service schemas into a combined Swagger UI.
