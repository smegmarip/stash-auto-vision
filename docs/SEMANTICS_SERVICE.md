# Semantics Service Documentation

**Service:** Semantics Service
**Port:** 5004
**Path:** `/semantics`
**Status:** Phase 2 - STUB (Not Yet Implemented)
**Version:** 1.0.0

---

## Summary

The Semantics Service is a planned microservice that will provide CLIP-based scene understanding, semantic tagging, and zero-shot classification capabilities for video content. This service is currently stubbed and returns "not_implemented" status for all endpoints.

### Planned Features (Phase 2)

- **CLIP Integration:** Vision-language model (ViT-B/32 or ViT-L/14) for scene understanding
- **Scene Classification:** Automatic categorization (indoor/outdoor, setting types, content types)
- **Zero-Shot Tagging:** Custom text prompts for flexible classification
- **Semantic Search:** Natural language queries to find scenes ("find scenes with two people talking in a kitchen")
- **Scene Embeddings:** Multi-modal embeddings for similarity-based search
- **Custom Tag Lists:** User-defined classification categories

### Current Status

This service is a **stub implementation** awaiting Phase 2 development. All endpoints return HTTP 202 (Accepted) or 501 (Not Implemented) with appropriate status messages. The health check endpoint returns "healthy" with `implemented: false` to indicate stub status.

Phase 2 implementation is planned after Phase 1 (face recognition and scene detection) is complete. The service architecture is designed and API contracts are defined, but the core CLIP model integration and inference logic have not yet been implemented.

### Integration with Vision API

When implemented, the Semantics Service will integrate into the Vision API's multi-module analysis workflow. Clients can enable semantic analysis by setting `modules.semantics.enabled: true` in the `/vision/analyze` request. The service will process frames extracted by the Frame Server and return semantic tags, scene classifications, and embeddings for each frame or scene boundary.

---

## OpenAPI 3.0 Schema

```yaml
openapi: 3.0.3
info:
  title: Semantics Service API (Stubbed)
  description: CLIP-based scene understanding service - Phase 2 implementation pending
  version: 1.0.0
servers:
  - url: http://semantics-service:5004
    description: Internal Docker network

paths:
  /analyze:
    post:
      summary: Analyze scene semantics (STUB)
      description: |
        Submit frames or video for semantic analysis using CLIP.

        **CURRENT STATUS:** Returns "not_implemented" response.

        **PLANNED IMPLEMENTATION:** Will process frames with CLIP to generate:
        - Scene classifications (indoor/outdoor, setting types)
        - Zero-shot tags based on custom prompts
        - Multi-modal embeddings for similarity search
        - Action/activity recognition
      operationId: analyzeSemantics
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AnalyzeSemanticsRequest'
      responses:
        '202':
          description: Request acknowledged (stub returns not_implemented)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SemanticsJobResponse'
        '400':
          description: Invalid request parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /jobs/{job_id}/status:
    get:
      summary: Get analysis job status (STUB)
      description: Poll semantic analysis job status
      operationId: getJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Job status (stub always returns not_implemented)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SemanticsJobStatus'

  /jobs/{job_id}/results:
    get:
      summary: Get analysis results (STUB)
      description: Retrieve semantic analysis results
      operationId: getJobResults
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '501':
          description: Not implemented
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: object
                    properties:
                      message:
                        type: string
                        example: "Semantics analysis module is not yet implemented (Phase 2)"
                      type:
                        type: string
                        example: "NotImplementedError"

  /health:
    get:
      summary: Service health check
      operationId: healthCheck
      responses:
        '200':
          description: Service healthy (stub service)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

components:
  schemas:
    AnalyzeSemanticsRequest:
      type: object
      description: Request schema for semantic analysis (PLANNED)
      properties:
        video_path:
          type: string
          description: Absolute path to video file
          example: "/media/videos/scene.mp4"
        scene_id:
          type: string
          description: Scene identifier for reference
        frame_extraction_job_id:
          type: string
          description: Job ID from Frame Server (reuse extracted frames)
        parameters:
          $ref: '#/components/schemas/SemanticsParameters'

    SemanticsParameters:
      type: object
      description: Configuration parameters for semantic analysis (PLANNED)
      properties:
        model:
          type: string
          enum: [clip-vit-b-32, clip-vit-l-14]
          default: clip-vit-b-32
          description: CLIP model variant
        classification_tags:
          type: array
          items:
            type: string
          description: Predefined tags for zero-shot classification
          example: ["indoor", "outdoor", "kitchen", "bedroom", "conversation", "action"]
        custom_prompts:
          type: array
          items:
            type: string
          description: Custom text prompts for zero-shot classification
          example: ["two people talking", "intimate scene", "outdoor activity"]
        generate_embeddings:
          type: boolean
          default: true
          description: Generate multi-modal embeddings for similarity search
        min_confidence:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 0.5
          description: Minimum confidence threshold for tag assignment
        top_k_tags:
          type: integer
          minimum: 1
          maximum: 20
          default: 5
          description: Maximum number of tags to return per frame

    SemanticsJobResponse:
      type: object
      properties:
        job_id:
          type: string
          description: Unique job identifier
          example: "semantics-stub-1699459200"
        status:
          type: string
          enum: [not_implemented, queued, processing, completed, failed]
          description: Job status (stub always returns not_implemented)
          example: "not_implemented"
        message:
          type: string
          description: Status message
          example: "Semantics analysis module is not yet implemented (Phase 2)"
        created_at:
          type: string
          format: date-time
          description: Job creation timestamp

    SemanticsJobStatus:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [not_implemented, queued, processing, completed, failed]
          example: "not_implemented"
        message:
          type: string
          example: "Semantics analysis module is not yet implemented (Phase 2)"
        created_at:
          type: string
          format: date-time

    SemanticsResults:
      type: object
      description: Semantic analysis results (PLANNED SCHEMA - not yet returned)
      properties:
        job_id:
          type: string
        scene_id:
          type: string
        status:
          type: string
        frames:
          type: array
          items:
            $ref: '#/components/schemas/FrameSemantics'
        scene_summary:
          $ref: '#/components/schemas/SceneSemanticSummary'
        metadata:
          $ref: '#/components/schemas/SemanticsMetadata'

    FrameSemantics:
      type: object
      description: Per-frame semantic analysis (PLANNED)
      properties:
        frame_index:
          type: integer
        timestamp:
          type: number
          format: float
        tags:
          type: array
          items:
            $ref: '#/components/schemas/SemanticTag'
        embedding:
          type: array
          items:
            type: number
            format: float
          description: Multi-modal embedding (512-D for ViT-B/32)
        scene_classification:
          type: object
          properties:
            setting:
              type: string
              example: "indoor"
            location:
              type: string
              example: "kitchen"
            activity:
              type: string
              example: "conversation"

    SemanticTag:
      type: object
      properties:
        tag:
          type: string
          description: Tag label
          example: "two people talking"
        confidence:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          description: Confidence score from CLIP
        source:
          type: string
          enum: [predefined, custom_prompt, zero_shot]
          description: Tag source type

    SceneSemanticSummary:
      type: object
      description: Aggregated semantic summary across all frames (PLANNED)
      properties:
        dominant_tags:
          type: array
          items:
            type: string
          description: Most frequent tags across frames
        scene_type:
          type: string
          description: Overall scene classification
        primary_activities:
          type: array
          items:
            type: string
        setting_changes:
          type: array
          items:
            type: object
            properties:
              timestamp:
                type: number
              old_setting:
                type: string
              new_setting:
                type: string

    SemanticsMetadata:
      type: object
      properties:
        model:
          type: string
          example: "clip-vit-b-32"
        frames_analyzed:
          type: integer
        processing_time_seconds:
          type: number
          format: float
        gpu_used:
          type: boolean
        total_tags_generated:
          type: integer

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
          example: "healthy"
        service:
          type: string
          example: "semantics-service"
        version:
          type: string
          example: "1.0.0"
        implemented:
          type: boolean
          example: false
          description: Indicates if service is fully implemented
        phase:
          type: integer
          example: 2
          description: Implementation phase number
        message:
          type: string
          example: "Stub service - awaiting CLIP integration"

    ErrorResponse:
      type: object
      properties:
        error:
          type: string
        detail:
          type: string
```

---

## Functional Details

### Current Implementation: Stub Service

The Semantics Service currently consists of a minimal FastAPI application that:

1. **Accepts requests** at all planned endpoints
2. **Returns stub responses** indicating not_implemented status
3. **Provides health checks** with `implemented: false` flag
4. **Does not process** any actual semantic analysis

**Stub Endpoints:**

```python
# POST /analyze
# Returns: {"job_id": "semantics-stub-{timestamp}", "status": "not_implemented"}

# GET /jobs/{job_id}/status
# Returns: {"status": "not_implemented", "message": "...Phase 2..."}

# GET /jobs/{job_id}/results
# Returns: HTTP 501 with NotImplementedError

# GET /health
# Returns: {"status": "healthy", "implemented": false, "phase": 2}
```

### Planned Implementation (Phase 2)

When Phase 2 development begins, the service will be upgraded with full CLIP integration.

#### CLIP Model Integration

**Model Selection:**
- **Primary:** OpenAI CLIP (ViT-B/32) - 224x224 input, 512-D embeddings
- **Optional:** OpenCLIP (ViT-L/14) - Higher accuracy, 768-D embeddings
- **Library:** `transformers` (Hugging Face) or `open_clip` (OpenCLIP)

**GPU Acceleration:**
```python
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

#### Processing Pipeline

```
1. Receive Request
   ├── Accept video_path or frame_extraction_job_id
   └── Parse parameters (tags, prompts, model selection)

2. Frame Acquisition
   ├── IF frame_extraction_job_id provided:
   │   └── Retrieve frames from Frame Server results
   └── ELSE:
       └── Request frame extraction from Frame Server

3. CLIP Inference (Per Frame)
   ├── Preprocess frame: Resize to 224x224, normalize
   ├── Generate image embedding (512-D for ViT-B/32)
   ├── IF classification_tags provided:
   │   ├── Encode text prompts
   │   ├── Calculate image-text similarity scores
   │   └── Assign tags above min_confidence threshold
   └── IF custom_prompts provided:
       └── Zero-shot classification with user prompts

4. Result Aggregation
   ├── Collect per-frame tags and embeddings
   ├── Generate scene summary (dominant tags, activities)
   ├── Detect setting/scene changes
   └── Format results as JSON

5. Return Results
   └── Store in Redis, return job_id
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

#### Performance Targets (Planned)

**GPU Mode (NVIDIA RTX 3060):**
- **Inference Speed:** 50-100 FPS (ViT-B/32)
- **Memory Usage:** ~2GB VRAM
- **Batch Size:** 16-32 frames per batch

**CPU Mode (Development):**
- **Inference Speed:** 5-10 FPS
- **Memory Usage:** ~1GB RAM
- **Not recommended for production**

**Processing Time Example:**
- 10-minute video @ 2 FPS sampling = 1200 frames
- GPU processing: ~12-24 seconds
- CPU processing: ~2-4 minutes

#### Future Parameters

When implemented, the service will accept these configuration parameters:

**model** (string): CLIP model variant
- `clip-vit-b-32` (default) - 224px, 512-D embeddings
- `clip-vit-l-14` - 224px, 768-D embeddings, higher accuracy

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

### Configuration (Planned)

Environment variables for Phase 2 implementation:

```bash
# Model configuration
CLIP_MODEL=clip-vit-b-32          # Model variant
CLIP_DEVICE=cuda                  # cuda or cpu

# Processing
BATCH_SIZE=16                     # Frames per batch
MIN_CONFIDENCE=0.5                # Tag confidence threshold

# Storage
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600                    # Result cache TTL

# Logging
LOG_LEVEL=INFO
```

### Integration with Vision API (Planned)

Enable semantic analysis in multi-module requests:

```json
{
  "video_path": "/media/videos/scene.mp4",
  "scene_id": "12345",
  "modules": {
    "semantics": {
      "enabled": true,
      "parameters": {
        "model": "clip-vit-b-32",
        "classification_tags": ["indoor", "outdoor", "kitchen", "conversation"],
        "custom_prompts": ["two people talking"],
        "min_confidence": 0.5,
        "top_k_tags": 5
      }
    }
  }
}
```

Vision API will:
1. Extract frames via Frame Server
2. Submit frames to Semantics Service
3. Aggregate results with faces/scenes data
4. Return combined analysis

### Use Cases (Phase 2 Goals)

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

**Last Updated:** 2025-11-09
**Status:** Stub Service - Phase 2 Implementation Pending
