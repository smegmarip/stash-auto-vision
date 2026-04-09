# How to Use Stash Auto Vision

## Quick Start

```bash
# Start services
docker compose up -d

# Check health
curl http://localhost:5010/vision/health

# Submit job
curl -X POST http://localhost:5003/faces/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345"
  }'

# Check status
curl http://localhost:5003/faces/jobs/{job_id}/status

# Get results
curl http://localhost:5003/faces/jobs/{job_id}/results
```

## Face Detection

**Simple:**

```bash
curl -X POST http://localhost:5003/faces/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene.mp4",
    "source_id": "123"
  }'
```

**With options:**

```json
{
  "source": "/media/videos/scene.mp4",
  "source_id": "123",
  "parameters": {
    "min_confidence": 0.9,
    "max_faces": 50,
    "enable_clustering": true
  }
}
```

## Face Enhancement

**Enable in environment:**

```bash
# Edit .env
ENABLE_ENHANCEMENT=true
ENHANCEMENT_MODEL=gfpgan

# Restart
docker compose restart frame-server
```

**Use in requests:**

```json
{
  "source": "/media/videos/scene.mp4",
  "enhancement": {
    "enabled": true,
    "fidelity_weight": 0.7
  }
}
```

**Single frame:**

```bash
curl "http://localhost:5001/frames/extract-frame?video_path=/media/videos/scene.mp4&timestamp=1.0&enhance=1&fidelity_weight=0.7"
```

**Fidelity weight:** 0.0 (preserve original) to 1.0 (max enhancement). Default: 0.7
**Note:** First run downloads models (~500MB). Enhancement upscales 2x, some detail loss possible.

## Scene Detection

```bash
curl -X POST http://localhost:5010/vision/analyze/scenes \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene.mp4",
    "source_id": "123",
    "parameters": {
      "threshold": 27.0
    }
  }'
```

## Video Semantics

Tag classification, scene summary, and suggested title in one pipeline. Scene metadata (sprite URLs, promotional description, performers) is resolved from Stash via `source_id` — leave `source` empty to let the service fetch the video path from Stash as well.

> **Tip:** the classifier scores each tag by encoding `"{name}: {description}"`, so adding descriptions to your Stash taxonomy — especially for ambiguous or jargony tags — produces noticeably better precision than relying on tag names alone. Descriptions are optional (the classifier falls back to the bare name) but strongly recommended. See [Semantics Service → Requirements](SEMANTICS_SERVICE.md#requirements) for details.

```bash
curl -X POST http://localhost:5004/semantics/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "",
    "source_id": "12345",
    "parameters": {
      "frame_selection":  "sprite_sheet",
      "frames_per_scene": 16,
      "min_confidence":   0.75,
      "top_k_tags":       30
    }
  }'
```

The response payload contains the predicted tags, per-frame captions, the narrative `scene_summary`, and a `suggested_title` — all under the `semantics` key of the results document.

## Combined Analysis

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene.mp4",
    "source_id": "123",
    "modules": {
      "scenes": {"enabled": true},
      "faces": {"enabled": true}
    }
  }'
```

## Configuration

**Confidence Thresholds:**

```bash
# Environment defaults (in .env)
FACES_MIN_CONFIDENCE=0.9
SCENES_THRESHOLD=27.0

# Per-request override
{
  "parameters": {
    "face_min_confidence": 0.7  # Lower for difficult videos
  }
}
```

**GPU vs CPU:**

```bash
# GPU mode (.env)
OPENCV_DEVICE=cuda
INSIGHTFACE_DEVICE=cuda

# CPU mode (.env.cpu.example)
OPENCV_DEVICE=cpu
INSIGHTFACE_DEVICE=cpu
```

## Environment Variables

All variables below are set in `.env` (copy from `.env.example` or `.env.cpu.example`). Values marked _required_ have no safe default — you must set them for the stack to behave correctly. Per-request parameters override environment defaults where applicable.

### Paths & Docker

| Variable            | Description                                                             | Default                     |
| ------------------- | ----------------------------------------------------------------------- | --------------------------- |
| `SERVER_MEDIA_PATH` | Host path to your live video library (mounted at `/data` in containers) | `/mnt/user/movies`          |
| `TEST_MEDIA_PATH`   | Host path to test fixtures (mounted at `/media/videos`)                 | `/tests/data`               |
| `DOCKER_NETWORK`    | Shared Docker network name (must exist before `docker compose up`)      | `stash-auto-vision-network` |
| `DOCKER_RUNTIME`    | Container runtime: `nvidia` for GPU, `runc` for CPU                     | `nvidia`                    |

### Service Ports

| Variable                | Description                                   | Default |
| ----------------------- | --------------------------------------------- | ------- |
| `VISION_API_PORT`       | Vision API orchestrator (use `5010` on macOS) | `5010`  |
| `FRAME_SERVER_PORT`     | Frame server                                  | `5001`  |
| `SCENES_PORT`           | Scenes service                                | `5002`  |
| `FACES_PORT`            | Faces service                                 | `5003`  |
| `SEMANTICS_PORT`        | Semantics service                             | `5004`  |
| `OBJECTS_PORT`          | Objects service (stub)                        | `5005`  |
| `RESOURCE_MANAGER_PORT` | Resource manager                              | `5007`  |
| `SCHEMA_PORT`           | Schema service (aggregated Swagger UI)        | `5009`  |
| `JOBS_VIEWER_PORT`      | Jobs Viewer React UI                          | `5020`  |
| `REDIS_PORT`            | Redis                                         | `6379`  |

### Frame Server

| Variable                  | Description                                                                    | Default       |
| ------------------------- | ------------------------------------------------------------------------------ | ------------- |
| `FRAME_SERVER_DOCKERFILE` | `Dockerfile` (GPU) or `Dockerfile.cpu`                                         | `Dockerfile`  |
| `FRAME_EXTRACTION_METHOD` | Primary extractor: `opencv_cuda`, `opencv_cpu`, `pyav_hw`, `pyav_sw`, `ffmpeg` | `opencv_cuda` |
| `OPENCV_DEVICE`           | OpenCV backend: `cuda` or `cpu`                                                | `cuda`        |
| `ENABLE_FALLBACK`         | Auto-fallback chain opencv → pyav_hw → pyav_sw → ffmpeg                        | `true`        |
| `ENABLE_ENHANCEMENT`      | Enable CodeFormer/GFPGAN face enhancement                                      | `false`       |
| `ENHANCEMENT_MODEL`       | `codeformer` (recommended) or `gfpgan`                                         | `codeformer`  |
| `MAX_ENHANCEMENT_PIXELS`  | Max pixels per frame for enhancement (default = 1080p)                         | `2073600`     |
| `FRAME_TTL_HOURS`         | Hours before extracted frames are auto-cleaned                                 | `2`           |
| `FRAME_DIR`               | Frame storage directory inside the container                                   | `/tmp/frames` |
| `FRAME_THREAD_POOL_SIZE`  | Worker threads for parallel frame extraction                                   | `4`           |

### Scenes Service

| Variable            | Description                                                    | Default              |
| ------------------- | -------------------------------------------------------------- | -------------------- |
| `SCENES_DOCKERFILE` | `Dockerfile` (GPU) or `Dockerfile.cpu`                         | `Dockerfile`         |
| `SCENES_DETECTOR`   | Detector backend: `pyscenedetect_cuda` or `pyscenedetect_cpu`  | `pyscenedetect_cuda` |
| `SCENES_THRESHOLD`  | ContentDetector threshold (0.0-100.0), per-request overridable | `27.0`               |

### Faces Service

| Variable                            | Description                                                                                               | Default                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `FACES_DOCKERFILE`                  | `Dockerfile` (GPU) or `Dockerfile.cpu`                                                                    | `Dockerfile`                       |
| `INSIGHTFACE_MODEL`                 | Model pack: `buffalo_l` (recommended) / `buffalo_s`                                                       | `buffalo_l`                        |
| `INSIGHTFACE_DEVICE`                | Inference device: `cuda` or `cpu`                                                                         | `cuda`                             |
| `INSIGHTFACE_COMPUTE_TYPE`          | Precision: `float16` (GPU), `int8` (CPU)                                                                  | `float16`                          |
| `FACES_MIN_CONFIDENCE`              | Detection confidence threshold (0.0-1.0), per-request overridable                                         | `0.9`                              |
| `FACES_MIN_QUALITY`                 | Minimum quality to retain a face (0.0-1.0, `0.0` = no filtering)                                          | `0.0`                              |
| `FACES_ENHANCEMENT_QUALITY_TRIGGER` | Quality threshold below which enhancement is triggered                                                    | `0.5`                              |
| `FACES_HF_REPO`                     | HuggingFace repo hosting occlusion + IQA ONNX weights                                                     | `smegmarip/face-recognition`       |
| `FACES_HF_OCCLUSION_MODEL`          | Path within the repo for the occlusion classifier (**required** — missing = 503 on `/faces/health`)      | `models/occlusion_classifier.onnx` |
| `FACES_HF_TOPIQ_MODEL`              | Path within the repo for TOPIQ-NR IQA model (optional, best-effort)                                       | `models/topiq_nr.onnx`             |
| `FACES_HF_CLIPIQA_MODEL`            | Path within the repo for CLIP-IQA+ IQA model (optional, best-effort)                                      | `models/clipiqa_plus.onnx`         |
| `FACES_HF_TOKEN`                    | Optional HF token for gated / private forks                                                               | _(empty)_                          |
| `FACES_LOCAL_OCCLUSION_PATH`        | Absolute in-container path to pre-downloaded occlusion ONNX (bypasses HF entirely — for air-gapped use)   | _(empty)_                          |
| `FACES_LOCAL_TOPIQ_PATH`            | Absolute in-container path to pre-downloaded TOPIQ-NR ONNX                                                | _(empty)_                          |
| `FACES_LOCAL_CLIPIQA_PATH`          | Absolute in-container path to pre-downloaded CLIP-IQA+ ONNX                                               | _(empty)_                          |
| `FACES_MODEL_CACHE_DIR`             | Container-side cache directory — must match the `faces_models_cache` volume mount in `docker-compose.yml` | `/app/models`                      |

### Semantics Service — Classifier

| Variable                          | Description                                                      | Default                      |
| --------------------------------- | ---------------------------------------------------------------- | ---------------------------- |
| `SEMANTICS_DOCKERFILE`            | `Dockerfile` (GPU) or `Dockerfile.cpu`                           | `Dockerfile`                 |
| `CLASSIFIER_MODEL`                | Classifier variant: `text-only`, `vision`, or path to checkpoint | `text-only`                  |
| `CLASSIFIER_DEVICE`               | Classifier device: `cuda` or `cpu`                               | `cuda`                       |
| `SEMANTICS_MIN_CONFIDENCE`        | Tag score threshold (0.0-1.0), per-request overridable           | `0.75`                       |
| `SEMANTICS_TAG_ID`                | Stash root tag ID for taxonomy subtree (empty = full taxonomy)   | _(empty)_                    |
| `SEMANTICS_HF_REPO`               | HuggingFace repo hosting the trained classifier weights          | `smegmarip/tag-classifier`   |
| `SEMANTICS_HF_VISION_MODEL`       | Path within the HF repo for the vision-variant checkpoint        | `vision/best_model.pt`       |
| `SEMANTICS_HF_TEXT_MODEL`         | Path within the HF repo for the text-only checkpoint             | `text-only/best_model.pt`    |
| `SEMANTICS_HF_VISION_TAG_MAPPING` | Path within the HF repo for the vision-variant tag mapping       | `vision/tag_mapping.json`    |
| `SEMANTICS_HF_TEXT_TAG_MAPPING`   | Path within the HF repo for the text-only tag mapping            | `text-only/tag_mapping.json` |

### Semantics Service — Captioning & Summary

| Variable                       | Description                                                             | Default                          |
| ------------------------------ | ----------------------------------------------------------------------- | -------------------------------- |
| `SEMANTICS_LLM_MODEL`          | Llama model used for scene summary **and** suggested title generation   | `RedHatAI/Llama-3.1-8B-Instruct` |
| `SEMANTICS_LLM_DEVICE`         | Llama runtime device: `cuda` (recommended) or `cpu` (~30x slower on 8B) | `cuda`                           |
| `SEMANTICS_HF_TOKEN`           | HuggingFace token (optional, required for gated models)                 | _(empty)_                        |
| `SEMANTICS_MODEL_IDLE_TIMEOUT` | Seconds before idle JoyCaption/Llama are unloaded to free VRAM          | `300`                            |

### Semantics Service — Job Queue

| Variable                 | Description                                                            | Default      |
| ------------------------ | ---------------------------------------------------------------------- | ------------ |
| `SEMANTICS_JOB_LOCK_TTL` | Max seconds a single job may hold the worker lock                      | `3600`       |
| `SEMANTICS_WORKER_ID`    | Worker identifier (defaults to container hostname)                     | _(hostname)_ |
| `SEMANTICS_MEMORY_LIMIT` | Container RAM limit (Llama 3.1 8B bfloat16 + peak overhead needs ~40G) | `40G`        |

### Objects Service (Phase 4, stub)

| Variable                 | Description                                                     | Default        |
| ------------------------ | --------------------------------------------------------------- | -------------- |
| `OBJECTS_DOCKERFILE`     | `Dockerfile` (GPU) or `Dockerfile.cpu`                          | `Dockerfile`   |
| `YOLO_MODEL`             | YOLO-World variant: `yolo-world-m` (GPU) / `yolo-world-s` (CPU) | `yolo-world-m` |
| `YOLO_DEVICE`            | YOLO device: `cuda` or `cpu`                                    | `cuda`         |
| `OBJECTS_STUB_MODE`      | Keep the service stubbed (set `false` once Phase 4 lands)       | `true`         |
| `OBJECTS_MIN_CONFIDENCE` | YOLO confidence threshold (0.0-1.0), per-request overridable    | `0.5`          |

### Stash Integration

| Variable        | Description                                                            | Default                 |
| --------------- | ---------------------------------------------------------------------- | ----------------------- |
| `STASH_URL`     | Stash instance base URL (used for taxonomy and scene metadata fetches) | `http://localhost:9999` |
| `STASH_API_KEY` | Stash GraphQL API key (empty if the instance has no auth)              | _(empty)_               |

### Resource Manager

| Variable                      | Description                                               | Default      |
| ----------------------------- | --------------------------------------------------------- | ------------ |
| `RESOURCE_MANAGER_DOCKERFILE` | `Dockerfile` (GPU) or `Dockerfile.cpu`                    | `Dockerfile` |
| `TOTAL_VRAM_MB`               | Total VRAM advertised to the lease broker                 | `16384`      |
| `LEASE_DURATION_SECONDS`      | Maximum single-lease hold time                            | `600`        |
| `HEARTBEAT_TIMEOUT_SECONDS`   | Heartbeat interval before an abandoned lease is reclaimed | `60`         |

### Cache, Processing & Logging

| Variable          | Description                                                                   | Default      |
| ----------------- | ----------------------------------------------------------------------------- | ------------ |
| `CACHE_TTL`       | Result cache TTL in seconds (default ≈ 1 year for long job-history retention) | `31536000`   |
| `PROCESSING_MODE` | Pipeline execution mode: `sequential` (recommended) or `parallel`             | `sequential` |
| `LOG_LEVEL`       | Python log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`             | `INFO`       |

## Troubleshooting

**Service not starting:**

```bash
docker compose logs {service-name}
```

**Cache issues:**

```bash
docker exec -it vision-redis redis-cli KEYS "*"
docker exec -it vision-redis redis-cli FLUSHALL  # Clear cache
```

**Port conflicts (macOS):**

```bash
# Use port 5010 (5000 conflicts with AirPlay)
VISION_API_PORT=5010
```

**Enhancement not working:**

```bash
# Check if enabled
docker compose logs frame-server | grep -i enhancement

# First run downloads models (~500MB, ~40s on startup)
docker compose logs frame-server | grep -i "Downloading\|GFPGAN initialized"

# Service will fail to start if ENABLE_ENHANCEMENT=true but init fails
```

## API Documentation

Interactive combined docs (aggregated by schema-service): `http://localhost:5009/docs`

Per-service raw schemas: `http://localhost:500{1..5}/openapi.json`

Full API spec artifact: `openapi.yml` in the repo root (generated from the live services).
