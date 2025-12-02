# Objects Service Documentation

**Service:** Objects Service
**Port:** 5005
**Path:** `/objects/analyze`
**Status:** Phase 3 - STUB (Not Yet Implemented)
**Version:** 1.0.0

---

## Summary

The Objects Service is a planned microservice that will provide open-vocabulary object detection and classification for video content using YOLO-World. This service is currently implemented as a stub returning "not implemented" responses and awaits Phase 3 development.

### Current Status

**STUB SERVICE - Phase 3 (Not Implemented)**

The service is currently a placeholder that accepts requests and returns "not_implemented" status. All endpoints are functional for API integration testing but do not perform actual object detection.

### Planned Capabilities (Phase 3)

When implemented, the Objects Service will provide:

- **YOLO-World Integration:** Open-vocabulary object detection with custom categories
- **Bounding Box Detection:** Precise object localization with confidence scores
- **Custom Object Categories:** User-defined object classes for domain-specific detection
- **Zero-Shot Detection:** Detect objects without explicit training on those categories
- **Temporal Tracking:** Track objects across frames to aggregate results
- **Multi-Object Support:** Detect and classify multiple objects simultaneously

### Architecture

The Objects Service will integrate with the Vision API orchestrator to provide object detection as part of comprehensive video analysis workflows. It will operate asynchronously with job submission, status polling, and result retrieval patterns consistent with other services.

**Planned Integration:**

```
vision-api → objects-service (YOLO-World)
                 ↓
         Object detections with:
         - Object class labels
         - Bounding boxes (x, y, w, h)
         - Confidence scores
         - Temporal aggregation
```

---

## OpenAPI 3.0 Schema

```yaml
openapi: 3.0.3
info:
  title: Objects Service API (Stub)
  description: Object detection service - awaiting YOLO-World integration (Phase 3)
  version: 1.0.0
servers:
  - url: http://objects-service:5005
    description: Internal Docker network
  - url: http://localhost:5005
    description: Development access

paths:
  /objects/analyze:
    post:
      summary: Analyze video for objects (STUB)
      description: |
        Submit video for object detection analysis.

        **Current Status:** Returns "not_implemented" - no actual processing
        **Phase 3:** Will use YOLO-World for open-vocabulary object detection
      operationId: analyzeObjects
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AnalyzeObjectsRequest"
      responses:
        "202":
          description: Job accepted (stub response)
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ObjectsJobResponse"
        "400":
          description: Invalid request parameters
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /objects/jobs/{job_id}/status:
    get:
      summary: Get object detection job status (STUB)
      description: Returns "not_implemented" status for all job IDs
      operationId: getJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Job status (stub)
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ObjectsJobStatus"

  /objects/jobs/{job_id}/results:
    get:
      summary: Get object detection results (STUB)
      description: Returns 501 Not Implemented error
      operationId: getJobResults
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "501":
          description: Not implemented
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ErrorResponse"

  /objects/health:
    get:
      summary: Service health check
      description: Returns healthy status with implemented=false flag
      operationId: healthCheck
      responses:
        "200":
          description: Service health
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"

components:
  schemas:
    AnalyzeObjectsRequest:
      type: object
      required:
        - video_path
      properties:
        video_path:
          type: string
          description: Absolute path to video file
          example: "/media/videos/scene.mp4"
        source_id:
          type: string
          description: Optional scene identifier
        parameters:
          $ref: "#/components/schemas/ObjectsParameters"

    ObjectsParameters:
      type: object
      description: Parameters for object detection (planned for Phase 3)
      properties:
        model:
          type: string
          enum: [yolo-world-v1, yolo-world-v2]
          default: yolo-world-v2
          description: YOLO-World model variant
        confidence_threshold:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 0.25
          description: Minimum detection confidence
        nms_threshold:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 0.45
          description: Non-maximum suppression threshold
        object_categories:
          type: array
          items:
            type: string
          description: Custom object categories to detect
          example: ["person", "chair", "table", "lamp"]
        max_detections:
          type: integer
          minimum: 1
          default: 100
          description: Maximum detections per frame
        sampling_interval:
          type: number
          format: float
          default: 2.0
          description: Seconds between analyzed frames

    ObjectsJobResponse:
      type: object
      properties:
        job_id:
          type: string
          description: Job identifier (stub format)
          example: "objects-stub-1699564800"
        status:
          type: string
          enum: [not_implemented]
          description: Always "not_implemented" in stub mode
        message:
          type: string
          description: Implementation status message
        created_at:
          type: string
          format: date-time

    ObjectsJobStatus:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [not_implemented]
        message:
          type: string
        created_at:
          type: string
          format: date-time

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy]
          description: Service is running
        service:
          type: string
          example: "objects-service"
        version:
          type: string
          example: "1.0.0"
        implemented:
          type: boolean
          description: Always false (stub service)
          example: false
        phase:
          type: integer
          description: Implementation phase number
          example: 3
        message:
          type: string
          description: Implementation status
          example: "Stub service - awaiting YOLO-World integration"

    ErrorResponse:
      type: object
      properties:
        error:
          type: object
          properties:
            message:
              type: string
            type:
              type: string
```

---

## Functional Details

### Current Implementation (Stub)

The Objects Service is currently a minimal FastAPI application that:

1. **Accepts Requests:** All endpoints accept valid requests without validation errors
2. **Returns Stub Responses:** Returns "not_implemented" status for all operations
3. **Provides Health Check:** Health endpoint indicates service is running but not implemented
4. **Maintains API Contract:** Ensures downstream services can integrate without errors

**Stub Behavior:**

- POST `/objects/analyze` → Returns 202 with `status: "not_implemented"`
- GET `/objects/jobs/{job_id}/status` → Returns 200 with `status: "not_implemented"`
- GET `/objects/jobs/{job_id}/results` → Returns 501 Not Implemented error
- GET `/objects/health` → Returns 200 with `implemented: false`

### Planned Implementation (Phase 3)

#### YOLO-World Model Integration

**Model:** YOLO-World (Open-Vocabulary Object Detection)

- **Architecture:** Real-time object detector with text-based category prompting
- **Capability:** Detect arbitrary object categories without retraining
- **Performance:** 30+ FPS inference on GPU (RTX A4000)
- **Variants:** YOLO-World-v1 (medium), YOLO-World-v2 (improved accuracy)

**Key Advantages:**

- No need for fixed category lists
- User-defined object categories at runtime
- Zero-shot detection (detects objects never seen during training)
- Efficient inference suitable for video processing

#### Object Detection Pipeline

**Processing Flow (Planned):**

```
1. Video Access
   └── Read from shared volume or receive frame paths from frame-server

2. Frame Sampling
   ├── Use interval-based sampling (default: 2s)
   └── OR accept pre-extracted frames from frame-server

3. YOLO-World Inference
   ├── Load model with custom object categories
   ├── Run detection on each frame
   ├── Apply confidence threshold filtering
   └── Apply non-maximum suppression (NMS)

4. Bounding Box Extraction
   ├── Extract box coordinates (x, y, w, h)
   ├── Class labels and confidence scores
   └── Frame index and timestamp metadata

5. Temporal Aggregation
   ├── Track objects across frames
   ├── Merge detections of same object
   └── Calculate object persistence (frames present)

6. Result Assembly
   ├── Group by object class
   ├── Include all detections per object
   └── Return JSON response
```

#### Detection Output Format (Planned)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_id": "12345",
  "status": "completed",
  "objects": [
    {
      "object_id": "obj_0",
      "class": "person",
      "detections": [
        {
          "frame_index": 45,
          "timestamp": 1.5,
          "bbox": {
            "x": 120,
            "y": 80,
            "width": 100,
            "height": 200
          },
          "confidence": 0.92
        }
      ],
      "persistence": {
        "first_seen": 1.5,
        "last_seen": 15.3,
        "frames_present": 45,
        "total_frames": 100
      }
    },
    {
      "object_id": "obj_1",
      "class": "chair",
      "detections": [
        {
          "frame_index": 10,
          "timestamp": 0.33,
          "bbox": {
            "x": 300,
            "y": 250,
            "width": 80,
            "height": 120
          },
          "confidence": 0.87
        }
      ],
      "persistence": {
        "first_seen": 0.33,
        "last_seen": 30.0,
        "frames_present": 89,
        "total_frames": 100
      }
    }
  ],
  "metadata": {
    "total_frames": 100,
    "frames_analyzed": 100,
    "unique_objects": 2,
    "total_detections": 134,
    "processing_time_seconds": 45.2,
    "model": "yolo-world-v2"
  }
}
```

### Parameters (Planned)

#### Model Selection

- `yolo-world-v1`: Original model, faster inference
- `yolo-world-v2`: Improved accuracy, slightly slower

#### Confidence Threshold

- Range: 0.0 - 1.0
- Default: 0.25
- Higher values = fewer false positives, may miss objects
- Lower values = more detections, may include false positives

#### NMS Threshold

- Range: 0.0 - 1.0
- Default: 0.45
- Controls suppression of overlapping boxes
- Higher = keep more overlapping boxes
- Lower = more aggressive suppression

#### Custom Object Categories

- User-defined list of objects to detect
- Examples: ["person", "dog", "car", "tree"]
- Open vocabulary: can specify any object
- No retraining required

#### Sampling Interval

- Seconds between analyzed frames
- Default: 2.0
- Trade-off between accuracy and speed
- Shorter = more detections, slower processing
- Longer = fewer detections, faster processing

### Performance Targets (Phase 3)

**GPU Mode (RTX A4000):**

- Inference: 30+ FPS per frame
- Full video (5 min @ 2s interval): ~5-8 minutes processing
- Memory: ~4GB VRAM

**CPU Mode:**

- Inference: 5-10 FPS per frame
- Full video (5 min @ 2s interval): ~15-25 minutes processing
- Memory: ~2GB RAM

**Optimization Strategies:**

- Batch inference (process multiple frames together)
- GPU acceleration mandatory for production
- Frame-level parallelism where possible
- Reuse model weights across jobs

### Use Cases (Phase 3)

#### Content Classification

Automatically detect and tag scene content:

- Furniture: chair, table, couch, bed
- Location indicators: tree, car, building, ocean
- Props: phone, laptop, book, glass

#### Safety and Content Filtering

Detect objects for content moderation:

- Weapons, alcohol, drugs
- User-defined prohibited items
- Flag scenes for manual review

#### Custom Object Detection

Domain-specific object detection:

- Industry tools and equipment
- Specific clothing items
- Brand-specific products
- Architectural elements

#### Action Recognition Support

Combine with other modules for advanced understanding:

- Objects + faces → "person sitting on chair"
- Objects + scene boundaries → "cooking scene with utensils"
- Temporal object patterns → action recognition

### Integration with Vision API (Phase 3)

The Objects Service will integrate into multi-module workflows:

```json
{
  "modules": {
    "scenes": {
      "enabled": true
    },
    "faces": {
      "enabled": true
    },
    "objects": {
      "enabled": true,
      "parameters": {
        "object_categories": ["person", "furniture"],
        "confidence_threshold": 0.3
      }
    }
  }
}
```

**Sequential Processing:**

1. Scenes Service detects boundaries
2. Frame Server extracts frames
3. Faces Service detects people
4. Objects Service detects furniture/props
5. Vision API aggregates all results

### Configuration (Planned)

Environment variables:

```bash
# Model settings
YOLO_WORLD_MODEL=yolo-world-v2
YOLO_DEVICE=cuda                    # cuda or cpu

# Detection parameters
DEFAULT_CONFIDENCE=0.25
DEFAULT_NMS_THRESHOLD=0.45
MAX_DETECTIONS_PER_FRAME=100

# Performance
BATCH_SIZE=8                        # Frames per batch
GPU_MEMORY_FRACTION=0.5            # GPU memory limit

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
```

---

## Phase 3 Implementation Plan

**Duration:** 2-3 days

### Deliverables

1. **YOLO-World Integration:**

   - Load YOLO-World model (medium variant)
   - Configure CUDA providers for GPU acceleration
   - Implement category-based detection

2. **Detection Pipeline:**

   - Frame sampling strategy
   - Batch inference implementation
   - Bounding box extraction
   - Confidence filtering and NMS

3. **Temporal Aggregation:**

   - Track objects across frames
   - Calculate persistence metrics
   - Merge duplicate detections

4. **API Implementation:**

   - Replace stub endpoints with actual processing
   - Job queue integration with Redis
   - Progress tracking and status updates
   - Result storage and retrieval

5. **Testing:**
   - Unit tests for detection logic
   - Integration tests with Vision API
   - Performance benchmarks (FPS, memory)
   - Accuracy validation with test videos

### Dependencies

- YOLO-World model files (download on build)
- CUDA 12.3.2 runtime
- PyTorch with CUDA support
- OpenCV for frame handling

---

**Last Updated:** 2025-11-09
**Status:** Stub Service - Awaiting Phase 3 Implementation
**Next Steps:** Complete Phase 2 (Semantics Service) before starting Phase 3
