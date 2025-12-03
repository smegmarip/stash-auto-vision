# Semantics Service Documentation

**Service:** Semantics Service
**Port:** 5004
**Path:** `/semantics/analyze`
**Status:** Phase 2 - IMPLEMENTED
**Version:** 1.0.0

---

## Summary

The Semantics Service provides SigLIP-based scene understanding, semantic tagging, and zero-shot classification capabilities for video content. This service uses Google's SigLIP (Sigmoid Loss for Language-Image Pre-Training) model for high-quality vision-language understanding.

### Implemented Features

- **SigLIP Integration:** Google's siglip-base-patch16-224 model (768-D embeddings)
- **Zero-Shot Classification:** Custom text prompts for flexible scene classification
- **Scene Embeddings:** Multi-modal embeddings for similarity-based search
- **Custom Tag Lists:** User-defined classification categories
- **Batch Processing:** Configurable batch sizes for efficient GPU/CPU inference
- **Scene-Aware Analysis:** Integration with scenes-service for scene boundary detection
- **Content-Based Caching:** Redis-based caching with SHA-256 keys
- **Memory Management:** Explicit cleanup to prevent GPU memory crashes

### Scene Integration

The Semantics Service supports two methods for scene-aware analysis:

1. **Vision-API Orchestration (Primary):** When both `scenes` and `semantics` modules are enabled in vision-api, scene boundaries are automatically passed from scenes-service to semantics-service. This enables per-scene semantic summaries without duplicate scene detection.

2. **Standalone with scenes_job_id (Optional):** The service can be called directly with a `scenes_job_id` parameter to fetch pre-computed scene boundaries from scenes-service. This allows re-running semantics with different parameters without re-detecting scenes.

### Integration with Vision API

The Semantics Service integrates into the Vision API's multi-module analysis workflow. Clients can enable semantic analysis by setting `modules.semantics.enabled: true` in the `/vision/analyze` request. The service processes frames extracted by the Frame Server and returns semantic tags, scene classifications, and embeddings for each frame. When scenes are detected, the service automatically generates per-scene semantic summaries with dominant tags.

---

## OpenAPI Schema

> **Note:** The live OpenAPI schema is auto-generated from FastAPI at runtime via `/semantics/openapi.json`. The schema-service at port 5009 aggregates all service schemas into a combined specification.

**Access the live schema:**

```bash
# Single service schema
curl http://localhost:5004/openapi.json | jq .

# Combined schema (all services)
curl http://localhost:5009/schema/openapi.json | jq .

# Swagger UI
open http://localhost:5009/docs
```

---

## Functional Details

### Current Implementation: Fully Functional

The Semantics Service is a fully implemented service with SigLIP integration. It provides:

1. **Zero-shot classification** using Google's SigLIP model
2. **Scene-aware analysis** with integration to scenes-service
3. **Multi-modal embeddings** (768-D) for similarity search
4. **Content-based caching** via Redis
5. **Asynchronous job processing** with progress tracking

**Active Endpoints:**

```python
# POST /semantics/analyze
# Returns: {"job_id": "uuid", "status": "queued|processing|completed"}

# GET /semantics/jobs/{job_id}/status
# Returns: {"status": "...", "progress": 0.0-1.0, "stage": "..."}

# GET /semantics/jobs/{job_id}/results
# Returns: Full semantics results with frames, tags, embeddings, scene summaries

# GET /semantics/health
# Returns: {"status": "healthy", "implemented": true, "model": "google/siglip-base-patch16-224"}
```

### Implementation Details

#### SigLIP Model Integration

**Model:**

- **Current:** Google SigLIP (siglip-base-patch16-224) - 224x224 input, 768-D embeddings
- **Library:** `transformers` (Hugging Face)
- **Advantages:** Better zero-shot performance than CLIP, sigmoid loss for improved calibration

**GPU Acceleration:**

```python
import torch
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
```

#### Processing Pipeline

```
0. Fetch Scene Boundaries (Optional)
   ├── IF scenes_job_id provided:
   │   ├── Call scenes-service to get pre-computed boundaries
   │   └── Use for scene-aware aggregation
   └── ELSE IF scene_boundaries in parameters:
       └── Use provided boundaries (from vision-api orchestration)

1. Frame Extraction
   ├── IF frame_extraction_job_id provided:
   │   └── Retrieve frames from Frame Server results
   └── ELSE:
       └── Request frame extraction (sampling_interval default: 2.0s)

2. Load Frame Images
   └── Batch load frames from Frame Server

3. SigLIP Classification (Batched)
   ├── IF classification_tags or custom_prompts provided:
   │   ├── Process frames in batches (default: 32)
   │   ├── Generate image embeddings (768-D)
   │   ├── Encode text prompts
   │   ├── Calculate image-text similarity scores
   │   └── Assign tags above min_confidence threshold (default: 0.5)
   └── Return top_k tags per frame (default: 5)

4. Generate Embeddings (Optional)
   ├── IF generate_embeddings=true (default):
   │   └── Extract 768-D embeddings for similarity search
   └── Clean up tensors/PIL images to prevent memory leaks

5. Aggregate Scene Summaries (Optional)
   ├── IF scene_boundaries provided:
   │   ├── Group frames by scene
   │   ├── Calculate dominant tags per scene (frequency-based)
   │   ├── Compute average confidence per scene
   │   └── Return scene summaries with timestamps
   └── ELSE:
       └── Return frame-level results only

6. Return Results
   ├── Store in Redis with content-based cache key
   ├── Return job_id
   └── Clean up memory (gc.collect(), torch.cuda.empty_cache())
```

#### Zero-Shot Classification

**Predefined Tags Example:**

```python
classification_tags = [
    "indoor scene",
    "outdoor scene",
    "kitchen",
    "bedroom",
    "conversation between two people",
    "action scene",
    "intimate scene"
]

# CLIP generates embeddings for both image and text
image_embedding = model.get_image_features(frame)
text_embeddings = model.get_text_features(classification_tags)

# Cosine similarity determines best matches
similarities = cosine_similarity(image_embedding, text_embeddings)
top_tags = [(tag, score) for tag, score in zip(classification_tags, similarities) if score > 0.5]
```

**Custom Prompts Example:**

```python
custom_prompts = [
    "two people talking in a kitchen",
    "romantic dinner scene",
    "outdoor sports activity"
]

# Zero-shot inference with user-defined prompts
```

#### Scene Embeddings for Similarity Search

CLIP generates multi-modal embeddings that can be used for semantic similarity:

```python
# Generate 512-D embedding for each frame
frame_embedding = model.get_image_features(frame)

# Store embeddings for similarity search
# Later: Find similar scenes by embedding distance
from scipy.spatial.distance import cosine
similarity = 1 - cosine(embedding1, embedding2)
```

**Use Case:** "Find scenes visually similar to this reference scene"

#### Performance

**GPU Mode (NVIDIA RTX 3060+):**

- **Inference Speed:** 50-100 FPS (SigLIP base)
- **Memory Usage:** ~2GB VRAM
- **Batch Size:** 16-32 frames per batch

**CPU Mode (Development):**

- **Inference Speed:** 5-10 FPS
- **Memory Usage:** ~1GB RAM
- **Not recommended for production**

**Processing Time Example:**

- 10-minute video @ 2 FPS sampling = 300 frames
- GPU processing: ~3-6 seconds
- CPU processing: ~30-60 seconds

#### Parameters

The service accepts these configuration parameters:

**model** (string): Model variant

- `google/siglip-base-patch16-224` (default) - 224px, 768-D embeddings

**classification_tags** (array): Predefined tag list for classification

- Example: `["indoor", "outdoor", "kitchen", "conversation"]`

**custom_prompts** (array): User-defined text prompts for zero-shot

- Example: `["two people talking", "intimate scene"]`

**generate_embeddings** (boolean): Generate multi-modal embeddings

- Default: `true`
- Used for similarity search and clustering

**min_confidence** (float): Minimum confidence for tag assignment

- Range: 0.0 - 1.0
- Default: 0.5

**top_k_tags** (integer): Maximum tags per frame

- Range: 1 - 20
- Default: 5

### Configuration

Environment variables:

```bash
# Model configuration
SIGLIP_MODEL=google/siglip-base-patch16-224  # Model variant
SEMANTICS_DEVICE=cuda                         # cuda or cpu

# Processing
SEMANTICS_BATCH_SIZE=32            # Frames per batch
SEMANTICS_MIN_CONFIDENCE=0.5       # Tag confidence threshold

# Storage
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600                     # Result cache TTL

# Logging
LOG_LEVEL=INFO
```

### Integration with Vision API

Enable semantic analysis in multi-module requests:

**Example 1: Semantics Only**

```json
{
  "source": "/media/videos/scene.mp4",
  "source_id": "test_001",
  "modules": {
    "semantics": {
      "enabled": true,
      "parameters": {
        "classification_tags": ["indoor", "outdoor", "kitchen", "conversation"],
        "custom_prompts": ["two people talking"],
        "min_confidence": 0.5,
        "top_k_tags": 5
      }
    }
  }
}
```

**Example 2: Scenes + Semantics (Scene-Aware Analysis)**

```json
{
  "source": "/media/videos/scene.mp4",
  "source_id": "test_002",
  "modules": {
    "scenes": {
      "enabled": true,
      "parameters": {
        "threshold": 27.0
      }
    },
    "semantics": {
      "enabled": true,
      "parameters": {
        "classification_tags": [
          "indoor",
          "outdoor",
          "bedroom",
          "kitchen",
          "conversation",
          "action"
        ],
        "min_confidence": 0.5
      }
    }
  }
}
```

Vision API will:

1. Detect scene boundaries via Scenes Service
2. Pass boundaries to Semantics Service
3. Semantics generates per-scene semantic summaries
4. Return combined results with scene-level dominant tags

**Example 3: Standalone with scenes_job_id**

```bash
# First, detect scenes
curl -X POST http://localhost:5002/semantics/analyze \
  -d '{"source": "/media/videos/scene.mp4", "source_id": "test_003"}'
# Returns: {"job_id": "scenes-abc123"}

# Then, run semantics with scene boundaries
curl -X POST http://localhost:5004/semantics/analyze \
  -d '{
    "source": "/media/videos/scene.mp4",
    "source_id": "test_003",
    "scenes_job_id": "scenes-abc123",
    "parameters": {
      "classification_tags": ["bedroom", "kitchen", "office"],
      "min_confidence": 0.6
    }
  }'
```

### Use Cases

**Auto-Tagging:**

- Automatically classify scene settings (indoor/outdoor, location types)
- Detect activities and actions (conversation, sports, intimate scenes)
- Tag content types for organization

**Semantic Search:**

- Natural language queries: "find scenes with two people talking in a kitchen"
- Multi-modal search combining text and visual similarity
- Content-based recommendations

**Scene Similarity:**

- Find visually/semantically similar scenes
- Cluster scenes by content type
- Detect scene changes based on semantic shift

**Advanced Classification:**

- Custom taxonomy definition via prompts
- Zero-shot classification without training data
- Extensible tagging system

---

**Last Updated:** 2025-12-02
**Version:** 2.0.0
**Status:** Phase 2 Complete - Fully Implemented with Scene Integration
