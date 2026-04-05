# Semantics Service Documentation

**Service:** Semantics Service
**Port:** 5004
**Path:** `/semantics/analyze`
**Status:** Phase 3 - Tag Classifier
**Version:** 3.0.0

---

## Summary

The Semantics Service provides automated tag classification for video content using a trained multi-view bi-encoder classifier. The service replaces the previous SigLIP zero-shot and JoyCaption captioning pipelines with a unified pipeline that achieves 99.2% match rate on labeled tags.

### Pipeline

The service runs a three-stage pipeline for each video:

1. **Frame Extraction** -- Extract representative frames from the video (scene-based or interval sampling, with optional sharpest-frame selection)
2. **JoyCaption Beta-One Captioning** -- Generate per-frame natural language captions using the JoyCaption beta-one VLM
3. **Llama 3.1 8B Narrative Summary** -- Aggregate frame captions into a single narrative summary via an external LLM API
4. **Tag Classification** -- Run the multi-view bi-encoder classifier against the Stash taxonomy to produce scored tags

### Key Features

- **Trained Bi-Encoder Classifier:** Multi-view architecture trained on labeled data, 99.2% match rate
- **Taxonomy Pre-Loading:** Loads full tag taxonomy from Stash at startup via `STASH_URL` + `SEMANTICS_TAG_ID`
- **JoyCaption Beta-One:** Upgraded VLM for frame captioning (~8GB VRAM, loaded/unloaded per job)
- **LLM Narrative Summary:** Llama 3.1 8B via external API for scene-level summarization
- **Lightweight Classifier:** ~1.4GB VRAM, kept loaded between jobs
- **Scene-Aware Analysis:** Integration with scenes-service for scene boundary detection
- **Sharpest Frame Selection:** Laplacian variance-based quality filtering
- **Sprite Sheet Support:** Fast frame extraction from pre-generated sprites
- **Asynchronous Job Processing:** Submit, poll, retrieve pattern with progress tracking
- **Content-Based Caching:** Redis-based caching with SHA-256 keys

### VRAM Budget

| Component | VRAM | Lifecycle |
|-----------|------|-----------|
| JoyCaption beta-one | ~8GB | Loaded/unloaded per job |
| Llama 3.1 8B | External API | No local VRAM |
| Tag classifier | ~1.4GB | Kept loaded |

---

## API Endpoints

### Submit Analysis Job

```
POST /semantics/analyze
```

Submits an asynchronous tag classification job.

**Request Body (`AnalyzeSemanticsRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | Path to video file |
| `source_id` | string | yes | Unique identifier (e.g., Stash scene ID) |
| `job_id` | string | no | Custom job ID (auto-generated if omitted) |
| `scenes_job_id` | string | no | Pre-computed scene boundaries job ID |
| `frame_extraction_job_id` | string | no | Pre-computed frame extraction job ID |
| `custom_taxonomy` | array/url | no | Override taxonomy (inline array or URL) |
| `parameters` | SemanticsParameters | no | Processing parameters |

**SemanticsParameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | string | - | Classifier model variant |
| `min_confidence` | float | 0.75 | Minimum confidence for tag assignment |
| `top_k_tags` | int | 30 | Maximum tags returned |
| `generate_embeddings` | bool | false | Generate scene embeddings |
| `use_hierarchical_decoding` | bool | false | Use hierarchical taxonomy decoding |
| `frame_selection` | string | - | Frame selection method (scene_based, interval) |
| `frames_per_scene` | int | 16 | Frames to extract per scene |
| `sampling_interval` | float | - | Seconds between frames (interval mode) |
| `select_sharpest` | bool | false | Filter by sharpness |
| `sharpness_candidate_multiplier` | int | - | Candidates per final frame |
| `min_frame_quality` | float | - | Minimum quality threshold (0-1) |
| `use_quantization` | bool | false | Use quantized models |
| `details` | bool | false | Include detailed output |
| `sprite_vtt_url` | string | - | URL to sprite VTT file |
| `sprite_image_url` | string | - | URL to sprite grid image |
| `scene_boundaries` | array | - | Explicit scene boundaries |

**Example:**

```bash
curl -X POST http://localhost:5004/semantics/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "parameters": {
      "min_confidence": 0.75,
      "top_k_tags": 30,
      "frames_per_scene": 16,
      "frame_selection": "scene_based"
    }
  }'
```

**Response:**

```json
{
  "job_id": "semantics-550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Semantics job queued",
  "created_at": "2026-04-04T12:34:56.789Z",
  "cache_hit": false
}
```

### Get Job Status

```
GET /semantics/jobs/{job_id}/status
```

**Example:**

```bash
curl http://localhost:5004/semantics/jobs/semantics-550e8400.../status | jq .
```

**Response:**

```json
{
  "job_id": "semantics-550e8400...",
  "status": "processing",
  "progress": 0.65,
  "stage": "captioning",
  "message": "Captioning frame 10/16"
}
```

### Get Job Results

```
GET /semantics/jobs/{job_id}/results
```

**Example:**

```bash
curl http://localhost:5004/semantics/jobs/semantics-550e8400.../results | jq .
```

**Response:**

```json
{
  "job_id": "semantics-550e8400...",
  "source_id": "12345",
  "status": "completed",
  "tags": [
    {
      "tag_id": "123",
      "tag_name": "Indoor",
      "score": 0.95,
      "path": "Setting/Indoor",
      "decode_type": "hierarchical"
    },
    {
      "tag_id": "456",
      "tag_name": "Living Room",
      "score": 0.88,
      "path": "Setting/Indoor/Living Room",
      "decode_type": "hierarchical"
    }
  ],
  "frame_captions": [
    {
      "frame_index": 0,
      "timestamp": 5.0,
      "caption": "A woman sits on a beige couch in a warmly lit living room..."
    }
  ],
  "scene_summary": "A woman relaxes in a warmly decorated living room, seated on a beige couch near a coffee table. The scene is shot at eye level with natural afternoon light...",
  "scene_embedding": [0.012, -0.034, 0.078, "..."]
}
```

### Health Check

```
GET /semantics/health
```

**Example:**

```bash
curl http://localhost:5004/semantics/health | jq .
```

**Response:**

```json
{
  "status": "healthy",
  "classifier_loaded": true,
  "taxonomy_loaded": true,
  "tag_count": 245
}
```

---

## Processing Pipeline

```
1. Taxonomy Pre-Load (Startup)
   └── Fetch full tag tree from Stash via STASH_URL + SEMANTICS_TAG_ID
       └── Build classifier label space from taxonomy

2. Frame Extraction
   ├── IF frame_extraction_job_id provided:
   │   └── Retrieve frames from Frame Server results
   ├── IF sprite_vtt_url + sprite_image_url provided:
   │   └── Extract frames from sprite sheets (fastest)
   └── ELSE:
       └── Request frame extraction (scene_based or interval)
           └── Optional: select_sharpest filtering

3. JoyCaption Beta-One Captioning
   ├── Load JoyCaption model (~8GB VRAM)
   ├── Caption each extracted frame
   └── Unload model to free VRAM

4. Llama 3.1 8B Narrative Summary
   ├── Send frame captions to external LLM API
   └── Receive aggregated scene narrative

5. Tag Classification
   ├── Encode narrative + captions with bi-encoder
   ├── Score against taxonomy embeddings
   ├── Apply min_confidence threshold (default 0.75)
   ├── Return top_k_tags (default 30)
   └── Each tag includes: tag_id, tag_name, score, path, decode_type

6. Return Results
   ├── Store in Redis with content-based cache key
   └── Return job_id for polling
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLASSIFIER_MODEL` | - | Path or name of trained bi-encoder model |
| `CLASSIFIER_DEVICE` | `cuda` | Device for classifier (cuda/cpu) |
| `STASH_URL` | `http://localhost:9999` | Stash instance URL |
| `STASH_API_KEY` | - | Stash API key |
| `SEMANTICS_TAG_ID` | - | Root tag ID for taxonomy tree |
| `LLM_API_BASE` | - | External LLM API base URL |
| `LLM_API_KEY` | - | External LLM API key |
| `LLM_MODEL` | - | LLM model name (e.g., llama-3.1-8b) |
| `SEMANTICS_MIN_CONFIDENCE` | `0.75` | Default minimum confidence |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `LOG_LEVEL` | `INFO` | Logging level |

### OpenAPI Schema

The live OpenAPI schema is auto-generated from FastAPI at runtime.

```bash
# Single service schema
curl http://localhost:5004/openapi.json | jq .

# Combined schema (all services)
curl http://localhost:5009/schema/openapi.json | jq .

# Swagger UI
open http://localhost:5009/docs
```

---

## Integration

### With Vision API

Enable semantic analysis in multi-module requests:

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "modules": {
      "scenes": {"enabled": true},
      "semantics": {
        "enabled": true,
        "parameters": {
          "min_confidence": 0.75,
          "top_k_tags": 30,
          "frames_per_scene": 16
        }
      }
    }
  }'
```

### With Pre-Computed Scenes

```bash
# First, detect scenes
curl -X POST http://localhost:5002/scenes/detect \
  -H "Content-Type: application/json" \
  -d '{"source": "/media/videos/scene_12345.mp4", "source_id": "12345"}'
# Returns: {"job_id": "scenes-abc123"}

# Then, run semantics with scene boundaries
curl -X POST http://localhost:5004/semantics/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "scenes_job_id": "scenes-abc123",
    "parameters": {
      "min_confidence": 0.75,
      "frames_per_scene": 16
    }
  }'
```

### With Stash Taxonomy

The service pre-loads the tag taxonomy from Stash at startup. Configure `STASH_URL`, `STASH_API_KEY`, and `SEMANTICS_TAG_ID` in the environment. The taxonomy is used to build the classifier's label space -- tags are matched by ID so results map directly back to Stash.

To override the taxonomy per-request, pass `custom_taxonomy` in the request body (either an inline array of tag objects or a URL to fetch taxonomy JSON).

---

## Performance

### Estimated Processing Times (GPU)

| Content | Frames | Approx. Time |
|---------|--------|---------------|
| 5 min video | 16 | ~30 seconds |
| 10 min video | 16 | ~35 seconds |
| 30 min video | 16 | ~40 seconds |

Processing time is dominated by captioning (per-frame) and LLM summarization, not video length. More frames = more time.

### Memory Usage

| Component | VRAM |
|-----------|------|
| JoyCaption beta-one (per job) | ~8GB |
| Tag classifier (persistent) | ~1.4GB |
| Peak during captioning | ~9.4GB |

---

## Troubleshooting

### Taxonomy Not Loading

```bash
# Check Stash connectivity
curl $STASH_URL/graphql -H "ApiKey: $STASH_API_KEY" -d '{"query": "{ tags { count } }"}'

# Check service logs
docker logs vision-semantics-service
```

### Poor Tag Quality

- Increase `frames_per_scene` for more visual coverage
- Enable `select_sharpest` for better frame quality
- Lower `min_confidence` if tags are being filtered too aggressively
- Verify taxonomy has sufficient tag descriptions in Stash

### High VRAM Usage

- JoyCaption is loaded/unloaded per job -- peak is ~9.4GB during captioning
- If OOM, ensure no other GPU services are running concurrently
- Check `nvidia-smi` during processing

---

## Related Documentation

- [Resource Manager](RESOURCE_MANAGER.md)
- [Scenes Service](SCENES_SERVICE.md)
- [Frame Server](FRAME_SERVER.md)

---

**Last Updated:** 2026-04-04
**Version:** 3.0.0
**Status:** Phase 3 Complete - Tag Classifier Pipeline
