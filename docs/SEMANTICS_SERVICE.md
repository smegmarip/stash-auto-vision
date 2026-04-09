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

1. **Frame Extraction** -- Extract frames from sprite sheets (default), scene-based, or interval sampling
2. **JoyCaption Beta-One Captioning** -- Generate per-frame natural language captions using the JoyCaption beta-one VLM
3. **Llama 3.1 8B Narrative Summary** -- Aggregate frame captions into a single narrative summary via an external LLM API
4. **Tag Classification** -- Run the multi-view bi-encoder classifier against the Stash taxonomy to produce scored tags

### Key Features

- **Trained Bi-Encoder Classifier:** Multi-view architecture trained on labeled data, 99.2% match rate
- **Taxonomy Pre-Loading:** Loads full tag taxonomy from Stash at startup via `STASH_URL` + `SEMANTICS_TAG_ID`
- **JoyCaption Beta-One:** Upgraded VLM for frame captioning (~8GB VRAM, loaded/unloaded per job)
- **LLM Narrative Summary:** Llama 3.1 8B via external API for scene-level summarization
- **Lightweight Classifier:** ~1.4GB VRAM, kept loaded between jobs
- **Stash Scene Resolution:** Fetches scene metadata (sprites, details, performers) from Stash via `source_id`
- **Sprite Sheet Default:** Fast frame extraction from pre-generated sprites (default mode)
- **Sharpest Frame Selection:** Laplacian variance-based quality filtering (video frame fallback)
- **Scene-Aware Analysis:** Optional integration with scenes-service for scene boundary detection
- **Asynchronous Job Processing:** Submit, poll, retrieve pattern with progress tracking
- **Content-Based Caching:** Redis-based caching with SHA-256 keys

### Requirements

**Minimum taxonomy entry.** Every tag needs at least an `id`, a `name`, and its `parents` list. The taxonomy builder computes ancestor paths (`all_paths`) from parents automatically and tolerates missing `description` / `aliases` fields, so a skeletal Stash taxonomy will load and classify without error.

**Tag descriptions are optional but strongly recommended.** The classifier is a bi-encoder that builds each tag's text embedding from `"{name}: {description}"` when a description is present, and falls back to the bare `name` when it isn't. The fallback works — the classifier was trained on a mix of annotated and unannotated tags — but the embedding carries noticeably more semantic signal when a description is provided. Concretely, the relevant code in `train/model.py` is:

```python
name, desc = tag.get("name", ""), tag.get("description", "")
tag_texts[idx] = f"{name}: {desc}" if desc else name
```

**Where descriptions help most.** The accuracy gap between annotated and unannotated tags is largest for tags whose names are ambiguous — single-word adjectives, acronyms, site-specific jargon, or terms that mean different things in different domains. A tag literally named `Wet` with no description is a much weaker embedding than the same tag described as _"scenes featuring water, rain, or visibly wet surfaces."_ The 99.2% match-rate headline figure reflects the taxonomy as it was trained, annotations and all — your own precision on a sparsely-annotated taxonomy will be lower.

**The highest-leverage annotation effort** is to add short, factual descriptions to your ambiguous or jargony tags first, and leave self-explanatory tags (e.g. proper nouns, unambiguous category names) for later. A one-sentence description is usually enough.

**What about `aliases`?** The classifier does not currently consume the `aliases` field at inference time. Only `name`, `description`, and `all_paths` (derived from `parents`) feed the tag embedding. If you rely on aliases for human-facing search in Stash, keep them — they just don't affect classifier accuracy.

### VRAM Budget

| Component           | VRAM         | Lifecycle               |
| ------------------- | ------------ | ----------------------- |
| JoyCaption beta-one | ~8GB         | Loaded/unloaded per job |
| Llama 3.1 8B        | External API | No local VRAM           |
| Tag classifier      | ~1.4GB       | Kept loaded             |

---

## API Endpoints

### Submit Analysis Job

```
POST /semantics/analyze
```

Submits an asynchronous tag classification job.

**Request Body (`AnalyzeSemanticsRequest`):**

| Field                     | Type                | Required | Description                                                              |
| ------------------------- | ------------------- | -------- | ------------------------------------------------------------------------ |
| `source`                  | string              | no       | Video path or URL (resolved from Stash if empty)                         |
| `source_id`               | string              | yes      | Stash scene ID. Used to fetch scene data (sprites, details, performers). |
| `job_id`                  | string              | no       | Custom job ID (auto-generated if omitted)                                |
| `scenes_job_id`           | string              | no       | Pre-computed scene boundaries job ID                                     |
| `frame_extraction_job_id` | string              | no       | Pre-computed frame extraction job ID                                     |
| `custom_taxonomy`         | array/url           | no       | Override taxonomy (inline array or URL)                                  |
| `parameters`              | SemanticsParameters | no       | Processing parameters                                                    |

**SemanticsParameters:**

| Parameter                        | Type   | Default      | Description                                                          |
| -------------------------------- | ------ | ------------ | -------------------------------------------------------------------- |
| `model_variant`                  | string | text-only    | Classifier model variant (`vision`, `text-only`, or checkpoint path) |
| `min_confidence`                 | float  | 0.75         | Minimum confidence for tag assignment                                |
| `top_k_tags`                     | int    | 30           | Maximum tags returned                                                |
| `generate_embeddings`            | bool   | false        | Return 512-D scene embeddings                                        |
| `use_hierarchical_decoding`      | bool   | true         | Apply taxonomy-consistent post-processing                            |
| `frame_selection`                | string | sprite_sheet | Frame selection method (`sprite_sheet`, `scene_based`, `interval`)   |
| `frames_per_scene`               | int    | 16           | Frames to extract (classifier trained on 16)                         |
| `sampling_interval`              | float  | 2.0          | Seconds between frames (interval mode)                               |
| `select_sharpest`                | bool   | true         | Filter by sharpness (video frame modes)                              |
| `sharpness_candidate_multiplier` | int    | 3            | Extract N \* frames_per_scene candidates for sharpness selection     |
| `min_frame_quality`              | float  | 0.05         | Minimum quality threshold (0-1)                                      |
| `use_quantization`               | bool   | true         | Use 4-bit quantization for JoyCaption VLM                            |
| `details`                        | string | -            | Promotional/editorial description (overrides Stash data)             |
| `sprite_vtt_url`                 | string | -            | URL to sprite VTT file (overrides Stash data)                        |
| `sprite_image_url`               | string | -            | URL to sprite grid image (overrides Stash data)                      |
| `scene_boundaries`               | array  | -            | Pre-computed scene boundaries                                        |

**Example:**

```bash
curl -X POST http://localhost:5004/semantics/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "12345",
    "parameters": {
      "min_confidence": 0.75,
      "top_k_tags": 30
    }
  }'
```

Scene metadata (sprite URLs, promotional description, performer info) is automatically fetched from Stash via `source_id`. Parameters override Stash data when explicitly provided.

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

2. Scene Resolution
   └── Fetch scene data from Stash via findScene(source_id)
       ├── Sprite URLs (paths.sprite, paths.vtt)
       ├── Promotional description (details)
       ├── Performers (count, genders)
       ├── Video metadata (duration, resolution)
       └── Request parameters override Stash data

3. Frame Extraction
   ├── DEFAULT (sprite_sheet): Extract from Stash sprite sheets
   ├── IF scene_based: Request frame extraction via Frame Server
   └── IF interval: Sample at fixed interval via Frame Server
       └── Optional: select_sharpest filtering (video frame modes)

4. JoyCaption Beta-One Captioning
   ├── Load JoyCaption model (~8GB VRAM)
   ├── Caption each extracted frame
   └── Unload model to free VRAM

5. Llama 3.1 8B Narrative Summary
   ├── Send frame captions + scene metadata to external LLM API
   └── Receive aggregated scene narrative

6. Tag Classification
   ├── Encode captions + summary + promo with bi-encoder
   ├── Encode each tag as "{name}: {description}" (see Requirements section)
   ├── Score against taxonomy embeddings
   ├── Apply min_confidence threshold (default 0.75)
   ├── Return top_k_tags (default 30)
   └── Each tag includes: tag_id, tag_name, score, path, decode_type

6. Return Results
   ├── Store in Redis with content-based cache key
   └── Return job_id for polling
```

---

## Model Management

### Classifier Source

The tag classifier is a **custom-trained multi-view bi-encoder** hosted on HuggingFace Hub at **[smegmarip/tag-classifier](https://huggingface.co/smegmarip/tag-classifier)**. Training was done in a separate experiment repository; only inference code is shipped with this service (bundled under `semantics-service/train/`).

### Variants

The HF repo publishes two flavors of the classifier:

| Variant               | Backbone                 | Description                                                                                                                     |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `text-only` (default) | Text-only bi-encoder     | Uses JoyCaption frame captions + Llama 3.1 scene summary as the sole input. Smaller, faster, no vision tower at inference time. |
| `vision`              | Multi-view vision + text | Uses frame pixel features alongside captions. Larger, higher headroom on visual-signal tags.                                    |

Each variant in the HF repo contains two files:

- `<variant>/best_model.pt` — PyTorch checkpoint
- `<variant>/tag_mapping.json` — label-space mapping for the variant (optional; the service will warn and continue if it's missing)

### Auto-Download at Startup

On first use, `semantics-service/app/classifier.py` calls `huggingface_hub.hf_hub_download()` to fetch the configured variant into `MODEL_CACHE_DIR` (default: `semantics-service/models/`). Subsequent starts use the cached copy.

```
semantics-service/models/
├── text-only/
│   ├── best_model.pt
│   └── tag_mapping.json
└── vision/
    ├── best_model.pt
    └── tag_mapping.json
```

The download is skipped entirely if `CLASSIFIER_MODEL` is set to an absolute file path — see "Custom Checkpoints" below.

### HuggingFace Repo / File Overrides

Every HF source path is overridable via environment variables, so you can point the service at a forked repo or at different file layouts without rebuilding the image:

| Variable                          | Default                      | Purpose                                                                                          |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `SEMANTICS_HF_REPO`               | `smegmarip/tag-classifier`   | HuggingFace repo ID                                                                              |
| `SEMANTICS_HF_TEXT_MODEL`         | `text-only/best_model.pt`    | Filename of the text-only checkpoint within the repo                                             |
| `SEMANTICS_HF_TEXT_TAG_MAPPING`   | `text-only/tag_mapping.json` | Filename of the text-only tag mapping within the repo                                            |
| `SEMANTICS_HF_VISION_MODEL`       | `vision/best_model.pt`       | Filename of the vision checkpoint within the repo                                                |
| `SEMANTICS_HF_VISION_TAG_MAPPING` | `vision/tag_mapping.json`    | Filename of the vision tag mapping within the repo                                               |
| `SEMANTICS_HF_TOKEN`              | -                            | HuggingFace access token (mapped to `HF_TOKEN` inside the container; required for private repos) |
| `MODEL_CACHE_DIR`                 | `models`                     | Local cache directory for downloaded checkpoints                                                 |

### Custom Checkpoints

Instead of using the HF-published variants, you can point the service at a local checkpoint file. Two options:

1. **Global override** — set `CLASSIFIER_MODEL` to an absolute path on the mounted volume:

   ```bash
   CLASSIFIER_MODEL=/models/my-custom-classifier.pt
   ```

2. **Per-request override** — pass the `model_variant` parameter in the analyze request:

   ```json
   {
     "source_id": "12345",
     "parameters": {
       "model_variant": "/models/my-custom-classifier.pt"
     }
   }
   ```

Any value that is not `text-only` or `vision` is treated as a filesystem path and loaded directly, bypassing the HF download step entirely. The checkpoint must be compatible with the `MultiViewClassifier` architecture defined in `semantics-service/train/model.py`.

---

## Configuration

### Environment Variables

| Variable                       | Default                            | Description                                                    |
| ------------------------------ | ---------------------------------- | -------------------------------------------------------------- |
| `CLASSIFIER_MODEL`             | `text-only`                        | Classifier variant (`text-only`, `vision`, or checkpoint path) |
| `CLASSIFIER_DEVICE`            | `cuda`                             | Device for classifier (cuda/cpu)                               |
| `STASH_URL`                    | `http://host.docker.internal:9999` | Stash instance URL                                             |
| `STASH_API_KEY`                | -                                  | Stash API key                                                  |
| `SEMANTICS_TAG_ID`             | -                                  | Root tag ID for taxonomy tree                                  |
| `SEMANTICS_LLM_MODEL`          | `RedHatAI/Llama-3.1-8B-Instruct`   | Local LLM for scene summary generation                         |
| `SEMANTICS_LLM_DEVICE`         | `cpu`                              | Device for the summary model (`cpu` to avoid VRAM contention)  |
| `SEMANTICS_HF_TOKEN`           | -                                  | HuggingFace token (mapped to `HF_TOKEN` inside the container)  |
| `SEMANTICS_MODEL_IDLE_TIMEOUT` | `300`                              | Seconds before idle models are unloaded from memory            |
| `SEMANTICS_JOB_LOCK_TTL`       | `3600`                             | Maximum seconds a job can hold the active queue lock           |
| `SEMANTICS_WORKER_ID`          | auto                               | Stable worker identifier (used for crash recovery)             |
| `SEMANTICS_MIN_CONFIDENCE`     | `0.75`                             | Default minimum confidence threshold                           |
| `REDIS_URL`                    | `redis://vision-redis:6379/0`      | Redis connection URL                                           |
| `LOG_LEVEL`                    | `INFO`                             | Logging level                                                  |

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

| Content      | Frames | Approx. Time |
| ------------ | ------ | ------------ |
| 5 min video  | 16     | ~30 seconds  |
| 10 min video | 16     | ~35 seconds  |
| 30 min video | 16     | ~40 seconds  |

Processing time is dominated by captioning (per-frame) and LLM summarization, not video length. More frames = more time.

### Memory Usage

| Component                     | VRAM   |
| ----------------------------- | ------ |
| JoyCaption beta-one (per job) | ~8GB   |
| Tag classifier (persistent)   | ~1.4GB |
| Peak during captioning        | ~9.4GB |

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
