# Stash Auto Vision

**GPU-Accelerated Video Analysis Microservices for Stash**

A modular, high-performance video analysis platform providing face recognition, scene detection, semantic tag classification, and (planned) object detection for the Stash media organizer ecosystem.

---

## Features

### Phase 1-3 (Complete)

- **Face Recognition** — InsightFace buffalo_l (99.86% LFW accuracy, 512-D ArcFace embeddings)
- **Face Enhancement** — Optional CodeFormer/GFPGAN for low-quality detections, three-tier quality gate
- **Scene Detection** — PySceneDetect with CUDA-accelerated OpenCV backend
- **Frame Server** — Multi-method extraction (OpenCV CUDA/CPU, PyAV, FFmpeg) with sprite-sheet support
- **Tag Classification** — Trained multi-view bi-encoder (99.2% match rate) with JoyCaption beta-one captioning and Llama 3.1 8B narrative summaries
- **Taxonomy Auto-Load** — Classifier label space is pre-loaded from the Stash tag tree at startup
- **Content-Based Caching** — Redis-backed SHA-256 keys with automatic invalidation on video changes
- **Unified Orchestration** — Vision API rollup endpoint with cross-namespace job listing
- **Jobs Viewer** — React-based UI for browsing jobs and inspecting results
- **Aggregated API Docs** — Schema-service combines per-service OpenAPI into a single Swagger UI
- **Plugin Integration — [stash-auto-vision-tagging](https://github.com/smegmarip/stash-auto-vision-tagging)** fully integrated (hybrid Go RPC + JS Stash plugin); **[stash-compreface-plugin](https://github.com/smegmarip/stash-compreface-plugin)** partially integrated (video recognition via Vision Service, pending final testing)

### Phase 4+ (Planned)

- **Object Detection** — YOLO-World open-vocabulary detection (service currently stubbed)
- **Production Hardening** — Retry logic, metrics, stress testing
- **Compreface Plugin Finalization** — Complete end-to-end testing of the partially-integrated [stash-compreface-plugin](https://github.com/smegmarip/stash-compreface-plugin) video recognition path

---

## Quick Start

### Prerequisites

**Production (Unraid / Linux + NVIDIA):**

- NVIDIA GPU with **≥16 GB VRAM** when running semantics (RTX A4000 / RTX 3090 or better). **12 GB is workable without semantics** — JoyCaption beta-one alone peaks ~8 GB during captioning, and the stack needs headroom above that. The `resource-manager` service (see `docs/RESOURCE_MANAGER.md`) brokers GPU leases between services, so multi-service contention is managed rather than simultaneous, which makes the VRAM ceiling less strict in practice.
- CUDA 12.x drivers and the NVIDIA container runtime
- Docker & Docker Compose

**Development (Mac / CPU-only):**

- Docker Desktop 4.0+
- 16 GB+ RAM (semantics-service is effectively unusable in CPU mode — JoyCaption is too slow; scenes/faces work fine)

### Installation

```bash
# Clone repository
git clone https://github.com/smegmarip/stash-auto-vision.git
cd stash-auto-vision

# Create the shared Docker network
docker network create stash-auto-vision-network

# GPU Mode (Production)
cp .env.example .env
# Edit .env — set SERVER_MEDIA_PATH, STASH_URL, STASH_API_KEY, SEMANTICS_TAG_ID

# CPU Mode (Development)
cp .env.cpu.example .env
# Paths are pre-configured to ./tests/data

# Start services
docker-compose up -d

# Verify orchestrator + all downstream services
curl http://localhost:5010/vision/health | jq .

# Open the combined Swagger UI (aggregated by schema-service)
open http://localhost:5009/docs
```

### Quick Test (via Vision API)

The orchestrator is the intended entry point for most use cases — it coordinates scenes, faces, and semantics with a single request.

```bash
# Submit a scenes + faces job via the rollup endpoint
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/compound/faces-service/single_person_varied_conditions.mp4",
    "source_id": "quick_test_001",
    "modules": {
      "scenes": { "enabled": true,  "parameters": { "scene_threshold": 27.0 } },
      "faces":  { "enabled": true,  "parameters": { "face_min_confidence": 0.9, "max_faces": 50 } }
    }
  }' | jq .

# → { "job_id": "...", "status": "queued", "services_enabled": {...} }

# Poll and fetch
JOB_ID="<paste-job-id-from-response>"
curl "http://localhost:5010/vision/jobs/$JOB_ID/status"  | jq .
curl "http://localhost:5010/vision/jobs/$JOB_ID/results" | jq .
```

To enable tag classification add `"semantics": { "enabled": true }` to `modules`. To pull a tag taxonomy and scene metadata from Stash automatically, make sure `STASH_URL` / `STASH_API_KEY` / `SEMANTICS_TAG_ID` are configured in your `.env`.

---

## Architecture

```
┌──────────────┐
│  vision-api  │ :5010 ← Main orchestrator (rollup endpoint)
└──────┬───────┘
       │
       ├─► frame-server     :5001 (frame extraction, sprite parsing)
       ├─► scenes-service   :5002 (scene boundary detection)
       ├─► faces-service    :5003 (face recognition + enhancement)
       ├─► semantics-service:5004 (tag classifier + captioning + summary)
       └─► objects-service  :5005 (YOLO-World, stub — Phase 4)

┌──────────────┐
│ schema-svc   │ :5009 ← Aggregated Swagger UI at /docs
└──────────────┘
┌──────────────┐
│ jobs-viewer  │ :5020 ← React monitoring UI
└──────────────┘
┌──────────────┐
│resource-mgr  │ :5007 ← GPU lease orchestration
└──────────────┘
┌──────────────┐
│    redis     │ :6379 ← Job metadata + result cache
└──────────────┘
```

**11 microservices** (all coordinated via `docker-compose.yml`):

- **redis** — Job metadata and result cache
- **dependency-checker** — Start-up health orchestration
- **frame-server** — Frame extraction (OpenCV CUDA/CPU, PyAV, FFmpeg, sprite sheets)
- **scenes-service** — PySceneDetect scene boundary detection
- **faces-service** — InsightFace buffalo_l recognition + optional CodeFormer/GFPGAN enhancement
- **semantics-service** — Trained multi-view bi-encoder tag classifier + JoyCaption beta-one + Llama 3.1 8B summary
- **objects-service** — Stubbed (Phase 4, YOLO-World planned)
- **resource-manager** — GPU/VRAM lease orchestration
- **vision-api** — Rollup orchestrator and cross-namespace job listing
- **schema-service** — Aggregates per-service OpenAPI into a unified Swagger UI
- **jobs-viewer** — React UI for browsing jobs and inspecting results

---

## API Examples

The Vision API is the primary entry point. All requests use a nested `modules` configuration — see [`docs/VISION_API.md`](docs/VISION_API.md) for the full schema.

### Scenes + Faces

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "modules": {
      "scenes": { "enabled": true },
      "faces":  { "enabled": true, "parameters": { "face_min_confidence": 0.9, "max_faces": 50 } }
    }
  }'
```

### Semantics (sprite-sheet frame source, no video decoding)

Scene metadata (sprite URLs, details, performers) is resolved from Stash via `source_id`.

> **Tag descriptions matter.** The classifier builds each tag's text embedding from `"{name}: {description}"` and falls back to the bare name when no description is set. The fallback works, but descriptions — especially on ambiguous or jargony tags — noticeably improve precision. See [Semantics Service → Requirements](docs/SEMANTICS_SERVICE.md#requirements) for details.

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "12345",
    "source": "",
    "modules": {
      "semantics": {
        "enabled": true,
        "parameters": {
          "frame_selection": "sprite_sheet",
          "min_confidence":  0.75,
          "top_k_tags":      30
        }
      }
    }
  }'
```

### Check Status / Get Results

```bash
curl http://localhost:5010/vision/jobs/{job_id}/status
curl http://localhost:5010/vision/jobs/{job_id}/results
```

### Cross-Service Job Listing

```bash
# List completed semantics jobs in the last 24h
curl "http://localhost:5010/vision/jobs?service=semantics&status=completed&limit=10"

# Efficient count for pagination / UI badges
curl "http://localhost:5010/vision/jobs/count?status=processing"
```

Jobs from every service (vision, faces, scenes, semantics, objects) are aggregated into a single listing, deduplicated by cache key.

---

## Documentation

- **[How to Use](docs/HOW_TO_USE.md)** — Quick start, API examples, troubleshooting
- **[Environment Variables](docs/HOW_TO_USE.md#environment-variables)** — Full reference of every `.env` setting, grouped by service
- **[Vision API](docs/VISION_API.md)** — Orchestrator request shape, aggregated results, job listing
- **[Project Specification](CLAUDE.md)** — Architecture, status, development notes
- **[Frame Server](docs/FRAME_SERVER.md)** — Frame extraction, sprite parsing, on-demand serving
- **[Scenes Service](docs/SCENES_SERVICE.md)** — PySceneDetect algorithms, boundary detection
- **[Faces Service](docs/FACES_SERVICE.md)** — InsightFace, enhancement, quality gate
- **[Semantics Service](docs/SEMANTICS_SERVICE.md)** — Tag classifier pipeline, taxonomy loading
- **[Resource Manager](docs/RESOURCE_MANAGER.md)** — GPU lease orchestration
- **[Objects Service](docs/OBJECTS_SERVICE.md)** — YOLO-World integration (Phase 4, stubbed)
- **[Testing Guide](docs/TESTING.md)** — Test scenarios and results
- **Live OpenAPI (all services):** <http://localhost:5009/docs> (Swagger UI, aggregated)
- **Jobs Viewer:** <http://localhost:5020/>

---

## Performance

### GPU Mode (RTX A4000, 16 GB VRAM) — _estimates_

These numbers are educated guesses derived from the current architecture and the component models in use. They have not yet been measured end-to-end on production hardware — treat them as planning targets, not benchmarks.

| Operation                               | Estimated Performance  |
| --------------------------------------- | ---------------------- |
| Frame extraction (OpenCV CUDA)          | ~200-400 FPS           |
| Scene detection (PySceneDetect on CUDA) | ~300-800 FPS           |
| Face detection (InsightFace det_size=640) | ~25-40 FPS             |
| Face embedding (ArcFace 512-D)          | <10 ms / face          |
| Tag classifier (inference only)         | ~50-100 frames / second|
| JoyCaption beta-one (per frame)         | ~1-2 s / frame         |
| Llama 3.1 8B summary (external API)     | ~2-5 s / scene         |
| Full scenes + faces (5 min video)       | ~15-30 s               |
| Full scenes + faces + semantics (5 min) | ~60-120 s              |

**VRAM budget (concurrent):** faces-service ~4 GB · tag classifier ~1.4 GB (resident) · JoyCaption ~8 GB (loaded only during captioning) · peak ~9.4 GB during a semantics job.

### CPU Mode (Mac Development)

Measured on a Mac development environment. Semantics is effectively unusable in CPU mode — benchmarks below omit it.

| Operation        | Measured Performance |
| ---------------- | -------------------- |
| Frame extraction | ~14-60 FPS           |
| Scene detection  | ~150 FPS             |
| Face detection   | ~5-7 FPS             |

---

## Technology Stack

**ML Models:**

- **InsightFace buffalo_l** — Face detection + ArcFace 512-D embeddings
- **PySceneDetect** — Scene boundary detection (content/threshold/adaptive)
- **CodeFormer / GFPGAN** — Optional face enhancement
- **Trained multi-view bi-encoder** — Tag classifier (99.2% match rate)
- **JoyCaption beta-one** — Per-frame captioning VLM (~8 GB VRAM, loaded per job)
- **Llama 3.1 8B** — Narrative summary via external API
- **YOLO-World** — Object detection (Phase 4)

**Infrastructure:**

- **FastAPI** — All backend services
- **Redis** — Job metadata, result cache, content-based keys
- **Docker Compose** — 10-service orchestration
- **NVIDIA CUDA 12.3.2** — GPU runtime
- **OpenCV** — CUDA-accelerated frame extraction and scene analysis
- **PyAV** — FFmpeg bindings for robust frame decoding

---

## Configuration

### GPU Mode (`.env`)

```bash
SERVER_MEDIA_PATH=/mnt/user/movies
DOCKER_RUNTIME=nvidia
INSIGHTFACE_DEVICE=cuda
OPENCV_DEVICE=cuda

# Required for semantics taxonomy pre-loading
STASH_URL=http://your-stash:9999
STASH_API_KEY=<your-api-key>
SEMANTICS_TAG_ID=<root-tag-id>
```

### CPU Mode (`.env`)

```bash
SERVER_MEDIA_PATH=./tests/data
DOCKER_RUNTIME=runc
INSIGHTFACE_DEVICE=cpu
OPENCV_DEVICE=cpu
VISION_API_PORT=5010  # macOS: port 5000 conflicts with AirPlay
```

### Confidence Thresholds

Environment variables set defaults; they can be overridden per request. Parameter names are service-prefixed so they don't collide inside the Vision API rollup.

```bash
FACES_MIN_CONFIDENCE=0.9               # face_min_confidence parameter
FACES_MIN_QUALITY=0.0                  # face_min_quality parameter
FACES_ENHANCEMENT_QUALITY_TRIGGER=0.5  # enhancement.quality_trigger
SCENES_THRESHOLD=27.0                  # scene_threshold parameter
SEMANTICS_MIN_CONFIDENCE=0.75          # semantics min_confidence parameter
OBJECTS_MIN_CONFIDENCE=0.5             # Phase 4 (stub)
CLASSIFIER_MODEL=text-only             # Tag classifier variant
```

All environment variables are documented in `.env.example` / `.env.cpu.example` and in the per-service markdown under `docs/`.

---

## Development

```bash
# Start all services
docker-compose up -d

# Tail logs
docker-compose logs -f vision-api
docker-compose logs -f semantics-service

# Rebuild after changes
docker-compose up -d --build faces-service

# Run tests (service-side)
docker-compose exec vision-api pytest

# Inspect Redis
docker exec -it vision-redis redis-cli
```

---

## Integration

Two Stash plugins consume this platform.

### `stash-auto-vision-tagging` — **Fully integrated** ✅

Scene tag classification inside Stash is handled by the companion project **[stash-auto-vision-tagging](https://github.com/smegmarip/stash-auto-vision-tagging)** — a hybrid Go RPC + JavaScript Stash plugin that was split out of this repository. The plugin:

1. Adds a per-scene toolbar button and a **Batch Tag Scenes** task to Stash.
2. Submits scene classification jobs to `stash-auto-vision` — dual-host routing between the **Vision Rollup API** (`:5010`) and the **Semantics Service** (`:5004`), with automatic submit-time failover.
3. Polls each job to completion and shows live progress in the Stash UI.
4. Applies the classifier's tags back to the scene via Stash's GraphQL mutations, with configurable merge/replace policy and flat + recursive tag exclusion lists.
5. Optionally writes the classifier's `scene_summary` into the scene's `details` field when it's empty.

See the plugin's [README](https://github.com/smegmarip/stash-auto-vision-tagging/blob/main/README.md) and `CLAUDE.md` for installation, settings, and the full behavior spec.

### `stash-compreface-plugin` — **Partially integrated** 🔄

The **[stash-compreface-plugin](https://github.com/smegmarip/stash-compreface-plugin)** adds face recognition and performer synchronization on top of Compreface. Its video recognition path (scene face extraction, sprite processing, embedding-based matching, face enhancement) is wired to use this project's faces-service / frame-server as its Vision Service backend. Integration is functionally complete but **still pending end-to-end testing** — expect rough edges on the video-path features until that work lands.

See the [plugin README](https://github.com/smegmarip/stash-compreface-plugin/blob/main/README.md) for feature details and current status.

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Check service logs
docker-compose logs faces-service | grep -i cuda
docker-compose logs semantics-service | grep -i cuda
```

### Services Not Starting

```bash
# dependency-checker orchestrates startup order
docker-compose logs dependency-checker

# Manually check individual service health
curl http://localhost:5001/frames/health
curl http://localhost:5002/scenes/health
curl http://localhost:5003/faces/health
curl http://localhost:5004/semantics/health
curl http://localhost:5010/vision/health
```

### Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Semantics is the heaviest consumer; if it's OOMing, ensure no other
# GPU services are concurrent and that the resource-manager is granting leases.
docker-compose logs resource-manager
```

### Cache Not Working

```bash
docker exec -it vision-redis redis-cli KEYS "*"
```

### Port Conflicts (macOS)

- Port 5000 conflicts with AirPlay Receiver on macOS.
- Solution: `VISION_API_PORT=5010` is already configured in `.env.cpu.example`.

---

## Roadmap

### Phase 1: Core Services ✅

- [x] Frame server with GPU-accelerated extraction
- [x] Scenes service (PySceneDetect)
- [x] Faces service (InsightFace + CodeFormer/GFPGAN enhancement)
- [x] Vision API orchestrator with unified job listing
- [x] End-to-end testing on Charades + CMU datasets

### Phase 2 & 3: Semantic Analysis ✅

- [x] Frame captioning (JoyCaption beta-one VLM)
- [x] Narrative summary (Llama 3.1 8B via external API)
- [x] Trained multi-view bi-encoder tag classifier (99.2% match rate)
- [x] Taxonomy pre-loading from Stash
- [x] Jobs Viewer UI with per-service filtering and semantics rendering
- [x] Schema-service aggregated Swagger UI
- [x] `stash-auto-vision-tagging` plugin integration (fully complete)
- [~] `stash-compreface-plugin` integration (video path wired, testing pending)

### Phase 4: Object Detection 🔄

- [ ] YOLO-World integration (medium variant)
- [ ] Open-vocabulary detection with custom categories
- [ ] Object tracking across frames

### Phase 5: Production Hardening 🔄

- [ ] Retry logic with exponential backoff
- [ ] Graceful degradation on partial service failure
- [ ] Structured logging and Prometheus metrics
- [ ] Stress testing with 10+ concurrent jobs

### Phase 6: Compreface Plugin Finalization 🔄

- [ ] End-to-end validation of the `stash-compreface-plugin` video recognition path
- [ ] Real-world benchmarks against live Stash instances

---

## Contributing

(TBD — contribution guidelines)

---

## License

(TBD — license information)

---

## Credits

**Technology:**

- [InsightFace](https://github.com/deepinsight/insightface) — Face recognition
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — Scene detection
- [CodeFormer](https://github.com/sczhou/CodeFormer) / [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face enhancement
- [JoyCaption](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) — Frame captioning VLM
- [Llama 3.1](https://llama.meta.com/) — Narrative summary LLM
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) — Object detection (planned)
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework

**Ecosystem:**

- [Stash](https://github.com/stashapp/stash) — Media organizer
- [stash-auto-vision-tagging](https://github.com/smegmarip/stash-auto-vision-tagging) — Tag classification Stash plugin (fully integrated)
- [stash-compreface-plugin](https://github.com/smegmarip/stash-compreface-plugin) — Face recognition Stash plugin (partially integrated)

---

**Status:** Phase 1-3 Complete — Scenes, Faces, and Semantic Tag Classification
**Version:** 3.0.0
**Last Updated:** 2026-04-08
