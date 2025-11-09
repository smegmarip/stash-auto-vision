# Faces Service Documentation

**Service:** Faces Service
**Port:** 5003
**Status:** Phase 1 - Implemented
**Version:** 1.0.0

---

## Summary

The Faces Service provides high-accuracy face detection and recognition using InsightFace's buffalo_l model. It detects faces in video frames, generates 512-dimensional ArcFace embeddings, performs quality scoring and pose estimation, and clusters similar faces through embedding similarity analysis.

The service integrates with the Frame Server for frame extraction and uses RetinaFace for detection and ArcFace for embedding generation. It supports content-based caching, async job processing, and automatic face deduplication via cosine similarity clustering.

### Key Features

- **High-Accuracy Detection:** InsightFace buffalo_l model with 99.86% LFW accuracy
- **512-D ArcFace Embeddings:** State-of-the-art face recognition vectors
- **Quality Scoring:** Multi-factor assessment (pose, size, confidence)
- **Face Deduplication:** Cosine similarity clustering (default threshold: 0.6)
- **Pose Estimation:** front, left, right, front-rotate-left, front-rotate-right
- **Smart Caching:** Content-based keys with automatic invalidation

### Processing Pipeline

1. Request frames from Frame Server (sampling_interval: 2.0s default)
2. Detect faces per frame using RetinaFace (min_confidence: 0.9)
3. Generate 512-D ArcFace embeddings
4. Score quality (pose, size, confidence)
5. Cluster faces by cosine similarity (threshold: 0.6)
6. Return unique faces with representative detections

---

## OpenAPI 3.0 Schema

```yaml
openapi: 3.0.3
info:
  title: Faces Service API
  version: 1.0.0
servers:
  - url: http://faces-service:5003

paths:
  /analyze:
    post:
      summary: Submit face analysis job
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [video_path]
              properties:
                video_path:
                  type: string
                  description: Absolute path to video file
                scene_id:
                  type: string
                job_id:
                  type: string
                parameters:
                  $ref: '#/components/schemas/FacesParameters'
      responses:
        '202':
          description: Job submitted
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: {type: string}
                  status: {type: string, enum: [queued]}
                  created_at: {type: string, format: date-time}

  /jobs/{job_id}/status:
    get:
      summary: Get job status
      parameters:
        - name: job_id
          in: path
          required: true
          schema: {type: string}
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: {type: string}
                  status: {type: string, enum: [queued, processing, completed, failed]}
                  progress: {type: number, minimum: 0, maximum: 1}
                  stage: {type: string}
                  message: {type: string}
                  error: {type: string}
                  result_summary:
                    type: object
                    properties:
                      unique_faces: {type: integer}
                      total_detections: {type: integer}
                      frames_processed: {type: integer}
                      processing_time_seconds: {type: number}

  /jobs/{job_id}/results:
    get:
      summary: Get analysis results
      parameters:
        - name: job_id
          in: path
          required: true
          schema: {type: string}
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: {type: string}
                  scene_id: {type: string}
                  status: {type: string}
                  faces:
                    type: array
                    items:
                      $ref: '#/components/schemas/Face'
                  metadata:
                    type: object
                    properties:
                      video_path: {type: string}
                      total_frames: {type: integer}
                      frames_processed: {type: integer}
                      unique_faces: {type: integer}
                      total_detections: {type: integer}
                      processing_time_seconds: {type: number}
                      method: {type: string}
                      model: {type: string}

  /health:
    get:
      summary: Service health check
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: {type: string}
                  service: {type: string}
                  version: {type: string}
                  model: {type: string}
                  gpu_available: {type: boolean}
                  active_jobs: {type: integer}
                  cache_size_mb: {type: number}

components:
  schemas:
    FacesParameters:
      type: object
      properties:
        min_confidence:
          type: number
          default: 0.9
          minimum: 0
          maximum: 1
          description: Minimum detection confidence
        max_faces:
          type: integer
          default: 50
          minimum: 1
          description: Maximum unique faces to extract
        sampling_interval:
          type: number
          default: 2.0
          description: Frame sampling interval (seconds)
        use_sprites:
          type: boolean
          default: false
        sprite_vtt_url:
          type: string
        sprite_image_url:
          type: string
        enable_deduplication:
          type: boolean
          default: true
          description: Cluster faces by embedding similarity
        embedding_similarity_threshold:
          type: number
          default: 0.6
          minimum: 0
          maximum: 1
          description: Cosine similarity threshold for clustering
        scene_boundaries:
          type: array
          items:
            type: object
            properties:
              start_timestamp: {type: number}
              end_timestamp: {type: number}
              duration: {type: number}
        cache_duration:
          type: integer
          default: 3600
          description: Cache TTL (seconds)

    Face:
      type: object
      properties:
        face_id:
          type: string
          description: Unique identifier for this face cluster
        embedding:
          type: array
          items: {type: number}
          minItems: 512
          maxItems: 512
          description: 512-D ArcFace embedding
        demographics:
          type: object
          properties:
            age: {type: integer}
            gender: {type: string, enum: [M, F]}
            emotion: {type: string}
        detections:
          type: array
          items:
            $ref: '#/components/schemas/Detection'
        representative_detection:
          $ref: '#/components/schemas/Detection'

    Detection:
      type: object
      properties:
        frame_index: {type: integer}
        timestamp: {type: number}
        bbox:
          type: object
          properties:
            x_min: {type: integer}
            y_min: {type: integer}
            x_max: {type: integer}
            y_max: {type: integer}
        confidence: {type: number, minimum: 0, maximum: 1}
        quality_score: {type: number, minimum: 0, maximum: 1}
        pose:
          type: string
          enum: [front, left, right, front-rotate-left, front-rotate-right]
        landmarks:
          type: object
          properties:
            left_eye: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
            right_eye: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
            nose: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
            mouth_left: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
            mouth_right: {type: array, items: {type: integer}, minItems: 2, maxItems: 2}
```

---

## Functional Details

### InsightFace Model

**Model:** buffalo_l
**Accuracy:** 99.86% on LFW (Labeled Faces in the Wild)
**Embedding:** 512-D ArcFace normalized vectors
**Detector:** RetinaFace (multi-scale, det_size: 640x640)
**Recognition:** ArcFace (additive angular margin loss)
**Optional:** Age and gender estimation

**Providers:**
- GPU Mode: CUDAExecutionProvider (ctx_id=0)
- CPU Mode: CPUExecutionProvider (ctx_id=-1)

### Detection Pipeline

**Frame Acquisition:**
- Requests frames from Frame Server at sampling_interval (default 2.0s)
- Supports interval, timestamp, scene-based sampling
- Optional sprite sheet parsing

**Face Detection:**
- RetinaFace multi-scale detection (640x640)
- Filter by min_confidence (default 0.9)
- Extract bounding boxes and 5-point landmarks

**Embedding Generation:**
- 512-D ArcFace embeddings (L2-normalized)
- Extract facial landmarks (eyes, nose, mouth)
- Optional demographics (age, gender)

**Quality Scoring:**
- Base: detection confidence
- Size factor: face area 10-30% of image is ideal
- Pose factor: front=1.0, rotated=0.9, profile=0.8
- Range: 0.0 - 1.0

**Pose Estimation:**
- Analyze landmark geometry (eye alignment, nose position)
- Classes: front, left, right, front-rotate-left, front-rotate-right
- Based on eye angle and nose-to-eye offset

**Face Clustering:**
- Calculate cosine similarity between embeddings
- Cluster if similarity >= threshold (default 0.6)
- Select representative (highest quality_score)

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| min_confidence | 0.9 | 0.0-1.0 | Detection threshold (0.7 for challenging, 0.9 for clean) |
| max_faces | 50 | 1+ | Max unique faces (10-20 typical, 50+ crowds) |
| sampling_interval | 2.0 | 0.5+ | Frame interval seconds (1.0-3.0 recommended) |
| enable_deduplication | true | bool | Cluster faces by similarity |
| embedding_similarity_threshold | 0.6 | 0.0-1.0 | Cosine threshold (0.5 loose, 0.6 balanced, 0.7 strict) |
| scene_boundaries | null | array | Scene timestamps for optimized sampling |
| cache_duration | 3600 | int | Redis TTL in seconds |

### Face Deduplication

**Cosine Similarity:**
```python
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
# similarity >= threshold → same person
```

**Clustering:**
1. Compare new face to all existing cluster representatives
2. If similarity >= threshold: add to cluster
3. Else: create new cluster
4. Track representative (highest quality_score)

**Quality Selection:**
- Representative = detection with max quality_score
- Factors: confidence × size_factor × pose_factor
- Used for face_id embedding

### Embedding Format

**Dimension:** 512 floats
**Normalization:** L2-normalized by InsightFace
**Encoding:** JSON array

**Example:**
```json
{
  "face_id": "face_0",
  "embedding": [-0.0364, -0.0076, 0.0671, ..., 0.0823]  // 512 values
}
```

**Usage:**
- Cosine similarity for clustering
- Vector database for large-scale matching
- Compreface API for subject recognition

### Performance Benchmarks

**CPU Mode (macOS Development):**
- Throughput: 2.5 FPS
- Test: 60s video, 60 frames in 23.68s
- Memory: ~600 MB

**GPU Mode (Expected - NVIDIA RTX A4000):**
- Throughput: 10-15 FPS (estimated)
- Test: 60s video, 60 frames in 4-6s
- Memory: ~1.5 GB VRAM
- Speedup: 4-6x vs CPU

**Comparison:**
| Metric | InsightFace buffalo_l | dlib (Legacy) |
|--------|----------------------|---------------|
| Accuracy | 99.86% LFW | 99.38% LFW |
| Speed (GPU) | 10-15 FPS | 0.5-1 FPS |
| Embedding | 512-D ArcFace | 128-D FaceNet |
| Status | Production | Research |

### Caching Strategy

**Cache Key Generation:**
```python
import hashlib, os, json

mtime = os.path.getmtime(video_path)
params_str = json.dumps(params, sort_keys=True)
cache_str = f"{video_path}:{mtime}:faces:{params_str}"
cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
```

**Redis Structure:**
```
faces:job:{job_id}:metadata       # Job status, progress
faces:job:{job_id}:result         # Job results
faces:cache:{cache_key}:job_id    # Cache lookup
```

**Invalidation:**
- File modification changes mtime → new cache_key
- Old entries expire after TTL (default 3600s)

### Configuration

```bash
# Model
INSIGHTFACE_MODEL=buffalo_l      # buffalo_l, buffalo_s, buffalo_sc
INSIGHTFACE_DEVICE=cuda          # cuda or cpu

# Integration
FRAME_SERVER_URL=http://frame-server:5001

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
```

### Vision API Integration

**Sequential Processing:**
1. vision-api → scenes-service (detect boundaries)
2. vision-api → faces-service (with scene_boundaries)
3. vision-api aggregates results

**Scene-Aware Sampling:**
- Extract 3 frames per scene (start, middle, end)
- Focuses on transitions, skips static scenes

**Example Request:**
```json
{
  "video_path": "/media/videos/scene.mp4",
  "scene_id": "12345",
  "parameters": {
    "min_confidence": 0.9,
    "sampling_interval": 2.0,
    "enable_deduplication": true,
    "embedding_similarity_threshold": 0.6,
    "scene_boundaries": [
      {"start_timestamp": 0.0, "end_timestamp": 15.0},
      {"start_timestamp": 15.0, "end_timestamp": 30.0}
    ]
  }
}
```

### Performance Characteristics

| Configuration | FPS | 60s Video | Use Case |
|--------------|-----|-----------|----------|
| CPU Mode | 2.5 | 24s | Development |
| GPU Mode | 10-15 | 4-6s | Production |
| CPU + Sprites | 5-10 | 6-12s | Fallback |
| GPU + Scene Opt | 15-20 | 3-4s | Optimized |

**Memory:**
- CPU: ~600 MB (model + runtime)
- GPU: ~1.5 GB VRAM
- Per face: ~2 KB (512 floats)

**Scaling:**
- Single video: 1 worker
- Multiple videos: N workers, parallel jobs
- High-volume: Load balancer + pool

---

**Last Updated:** 2025-11-09
**Status:** Implemented and Tested
