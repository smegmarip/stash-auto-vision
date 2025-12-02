# Faces Service Documentation

**Service:** Faces Service
**Port:** 5003
**Status:** Phase 1 - Implemented
**Version:** 1.0.0

---

## Summary

The Faces Service provides high-accuracy face detection and recognition using InsightFace's buffalo_l model. It detects faces in video frames, generates 512-dimensional ArcFace embeddings, performs quality scoring and pose estimation, and clusters similar faces through embedding similarity analysis.

The service integrates with the Frame Server for frame extraction and uses RetinaFace for detection and ArcFace for embedding generation. It supports content-based caching, async job processing, and automatic face deduplication via cosine similarity clustering.

###Key Features

- **High-Accuracy Detection:** InsightFace buffalo_l model with 99.86% LFW accuracy
- **Multi-Size Detection:** Automatic det_size selection (320/640/1024) based on image dimensions
- **512-D ArcFace Embeddings:** State-of-the-art face recognition vectors
- **Quality Scoring:** TOPIQ/CLIP-IQA sharpness + size/pose/occlusion components
- **Face Enhancement:** Optional CodeFormer/GFPGAN enhancement for low-quality detections
- **Three-Tier Quality System:** Detection confidence, quality trigger, minimum quality filtering
- **Face Deduplication:** Cosine similarity clustering (default threshold: 0.6)
- **Pose Estimation:** front, left, right, front-rotate-left, front-rotate-right
- **Smart Caching:** Content-based keys with automatic invalidation

### Face Enhancement

The service supports optional face enhancement for low-quality detections using production-grade AI models.

#### Three-Tier Quality Gate System

1. **face_min_confidence** (0.0-1.0, default 0.7 CPU / 0.9 GPU)

   - Initial detection threshold
   - Only faces above this confidence are considered
   - Environment variable: `FACES_MIN_CONFIDENCE`

2. **enhancement.quality_trigger** (0.0-1.0, default 0.5)

   - Triggers enhancement when quality score falls below this threshold
   - Applied to high-confidence, low-quality faces
   - Environment variable: `FACES_ENHANCEMENT_QUALITY_TRIGGER`

3. **face_min_quality** (0.0-1.0, default 0.0)
   - Final filtering threshold after enhancement
   - Only faces meeting this quality are returned
   - Environment variable: `FACES_MIN_QUALITY`

#### Enhancement Workflow

```
1. Detect faces with InsightFace (confidence >= face_min_confidence)
2. Calculate quality scores for each detection
3. For faces with quality < enhancement.quality_trigger:
   a. Request enhanced frame from frame-server
   b. Re-run InsightFace detection on enhanced frame
   c. Update quality_score and confidence
4. Filter: Keep only faces with quality >= face_min_quality
5. Mark enhanced faces with enhanced=true flag in detection metadata
```

#### Enhancement Models

**CodeFormer (Recommended):**

- Production-grade quality comparable to commercial solutions (Nero)
- GPU: 10-15x faster than CPU (~10-15ms per face)
- Fidelity weight: 0.0-1.0 (0.25 for heavy enhancement, 0.5 balanced, 0.7 for preservation)

**GFPGAN:**

- Legacy option, may over-smooth details
- Faster but lower quality than CodeFormer

#### Enhancement Parameters

```json
{
  "parameters": {
    "min_confidence": 0.7,
    "min_quality": 0.4,
    "enhancement": {
      "enabled": true,
      "quality_trigger": 0.5,
      "model": "codeformer",
      "fidelity_weight": 0.5
    }
  }
}
```

#### Performance Characteristics

- **Enhancement Decision:** Automatic based on quality scores
- **Enhanced Flag:** `detection.enhanced = true` for enhanced faces
- **GPU Speedup:** 10-15x faster enhancement on GPU vs CPU
- **Frame Caching:** Enhanced frames cached to avoid re-processing
- **Batch Optimization:** Single frame enhancement for multiple faces

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
  /faces/analyze:
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
                source_id:
                  type: string
                job_id:
                  type: string
                parameters:
                  $ref: "#/components/schemas/FacesParameters"
      responses:
        "202":
          description: Job submitted
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: { type: string }
                  status: { type: string, enum: [queued] }
                  created_at: { type: string, format: date-time }

  /faces/jobs/{job_id}/status:
    get:
      summary: Get job status
      parameters:
        - name: job_id
          in: path
          required: true
          schema: { type: string }
      responses:
        "200":
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: { type: string }
                  status:
                    {
                      type: string,
                      enum: [queued, processing, completed, failed],
                    }
                  progress: { type: number, minimum: 0, maximum: 1 }
                  stage: { type: string }
                  message: { type: string }
                  error: { type: string }
                  result_summary:
                    type: object
                    properties:
                      unique_faces: { type: integer }
                      total_detections: { type: integer }
                      frames_processed: { type: integer }
                      processing_time_seconds: { type: number }

  /faces/jobs/{job_id}/results:
    get:
      summary: Get analysis results
      parameters:
        - name: job_id
          in: path
          required: true
          schema: { type: string }
      responses:
        "200":
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id: { type: string }
                  source_id: { type: string }
                  status: { type: string }
                  faces:
                    type: array
                    items:
                      $ref: "#/components/schemas/Face"
                  metadata:
                    type: object
                    properties:
                      video_path: { type: string }
                      total_frames: { type: integer }
                      frames_processed: { type: integer }
                      unique_faces: { type: integer }
                      total_detections: { type: integer }
                      processing_time_seconds: { type: number }
                      method: { type: string }
                      model: { type: string }

  /faces/health:
    get:
      summary: Service health check
      responses:
        "200":
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: { type: string }
                  service: { type: string }
                  version: { type: string }
                  model: { type: string }
                  gpu_available: { type: boolean }
                  active_jobs: { type: integer }
                  cache_size_mb: { type: number }

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
              start_timestamp: { type: number }
              end_timestamp: { type: number }
              duration: { type: number }
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
          items: { type: number }
          minItems: 512
          maxItems: 512
          description: 512-D ArcFace embedding
        demographics:
          type: object
          properties:
            age: { type: integer }
            gender: { type: string, enum: [M, F] }
            emotion: { type: string }
        detections:
          type: array
          items:
            $ref: "#/components/schemas/Detection"
        representative_detection:
          $ref: "#/components/schemas/Detection"

    Detection:
      type: object
      properties:
        frame_index: { type: integer }
        timestamp: { type: number }
        bbox:
          type: object
          properties:
            x_min: { type: integer }
            y_min: { type: integer }
            x_max: { type: integer }
            y_max: { type: integer }
        confidence: { type: number, minimum: 0, maximum: 1 }
        quality:
          type: object
          properties:
            composite: { type: number, minimum: 0, maximum: 1 }
            components:
              type: object
              properties:
                size: { type: number, minimum: 0, maximum: 1 }
                pose: { type: number, minimum: 0, maximum: 1 }
                occlusion: { type: number, minimum: 0, maximum: 1 }
                sharpness:
                  {
                    type: number,
                    minimum: 0,
                    maximum: 1,
                    description: "TOPIQ/CLIP-IQA score",
                  }
        pose:
          type: string
          enum: [front, left, right, front-rotate-left, front-rotate-right]
        landmarks:
          type: object
          properties:
            left_eye:
              {
                type: array,
                items: { type: integer },
                minItems: 2,
                maxItems: 2,
              }
            right_eye:
              {
                type: array,
                items: { type: integer },
                minItems: 2,
                maxItems: 2,
              }
            nose:
              {
                type: array,
                items: { type: integer },
                minItems: 2,
                maxItems: 2,
              }
            mouth_left:
              {
                type: array,
                items: { type: integer },
                minItems: 2,
                maxItems: 2,
              }
            mouth_right:
              {
                type: array,
                items: { type: integer },
                minItems: 2,
                maxItems: 2,
              }
        enhanced: { type: boolean, default: false }
        occlusion:
          type: object
          properties:
            occluded: { type: boolean }
            probability: { type: number, minimum: 0, maximum: 1 }
```

---

## Functional Details

### InsightFace Model

**Model:** buffalo_l
**Accuracy:** 99.86% on LFW (Labeled Faces in the Wild)
**Embedding:** 512-D ArcFace normalized vectors
**Detector:** RetinaFace (multi-scale adaptive det_size)
**Recognition:** ArcFace (additive angular margin loss)
**Optional:** Age and gender estimation

**Multi-Size Detection:**

- CPU: 2 instances (det_size 320, 640)
- GPU: 3 instances (det_size 320, 640, 1024)
- Auto-selection: <500px→320, 500-1500px→640, >1500px→1024
- Prevents upscaling failures on small images

**Providers:**

- GPU Mode: CUDAExecutionProvider (ctx_id=0)
- CPU Mode: CPUExecutionProvider (ctx_id=-1)

### Detection Pipeline

**Frame Acquisition:**

- Requests frames from Frame Server at sampling_interval (default 2.0s)
- Supports interval, timestamp, scene-based sampling
- Optional sprite sheet parsing

**Sprite Sheet Integration:**

- Supports ultra-fast face detection from pre-generated WebVTT sprite tiles
- Parameters: `use_sprites`, `sprite_vtt_url`, `sprite_image_url`
- Automatic extraction method selection: "sprites" when `use_sprites=true`
- Shared volume: `/tmp/sprites` mounted between frame-server and faces-service
- Enhancement skipped: Sprite tiles are pre-processed at low resolution
- Performance: 100+ FPS vs ~30 FPS for video extraction

**Face Detection:**

- RetinaFace multi-scale detection (640x640)
- Filter by min_confidence (default 0.9)
- Extract bounding boxes and 5-point landmarks

**Embedding Generation:**

- 512-D ArcFace embeddings (L2-normalized)
- Extract facial landmarks (eyes, nose, mouth)
- Optional demographics (age, gender)

**Quality Scoring:**

- **Composite Score:** 35% size + 20% pose + 20% occlusion + 25% sharpness
- **Size Component:** Min dimension (80px=0.25, 250px=1.0)
- **Pose Component:** Yaw/pitch angles (0°=1.0, 45°=0.25)
- **Occlusion Component:** ResNet18 classifier (non-occluded=1.0, occluded=0.25)
- **Sharpness Component:** TOPIQ-NR (primary), CLIP-IQA (fallback), Sobel (last resort)
- **IQA Method:** ONNX Runtime (IR v10), ~133ms/image
- **Range:** 0.0 - 1.0

**Pose Estimation:**

- **Primary Method:** Native InsightFace pose angles (pitch, yaw, roll in degrees)
  - front: |yaw| < 15° and |roll| < 15°
  - left: yaw < -15°
  - right: yaw > 15°
  - front-rotate-left: |yaw| < 15° and roll < -15°
  - front-rotate-right: |yaw| < 15° and roll > 15°
- **Fallback Method:** Geometric estimation from landmark positions (eye alignment, nose position)
- **Classes:** front, left, right, front-rotate-left, front-rotate-right

**Face Clustering:**

- Calculate cosine similarity between embeddings
- Cluster if similarity >= threshold (default 0.6)
- Select representative (highest quality_score)

### Parameters

| Parameter                      | Default    | Range   | Description                                              |
| ------------------------------ | ---------- | ------- | -------------------------------------------------------- |
| min_confidence                 | 0.9        | 0.0-1.0 | Detection threshold (0.7 for challenging, 0.9 for clean) |
| min_quality                    | 0.0        | 0.0-1.0 | Minimum quality score to keep (0.0 = no filtering)       |
| max_faces                      | 50         | 1+      | Max unique faces (10-20 typical, 50+ crowds)             |
| sampling_interval              | 2.0        | 0.5+    | Frame interval seconds (1.0-3.0 recommended)             |
| enable_deduplication           | true       | bool    | Cluster faces by similarity                              |
| embedding_similarity_threshold | 0.6        | 0.0-1.0 | Cosine threshold (0.5 loose, 0.6 balanced, 0.7 strict)   |
| scene_boundaries               | null       | array   | Scene timestamps for optimized sampling                  |
| cache_duration                 | 3600       | int     | Redis TTL in seconds                                     |
| enhancement.enabled            | false      | bool    | Enable face enhancement for low-quality detections       |
| enhancement.quality_trigger    | 0.5        | 0.0-1.0 | Trigger enhancement if quality below this threshold      |
| enhancement.model              | codeformer | string  | Enhancement model: "codeformer" or "gfpgan"              |
| enhancement.fidelity_weight    | 0.5        | 0.0-1.0 | Quality vs fidelity tradeoff (CodeFormer only)           |

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

# Quality Thresholds
FACES_MIN_CONFIDENCE=0.9         # Detection confidence threshold (0.7 CPU, 0.9 GPU)
FACES_MIN_QUALITY=0.0            # Minimum quality to keep (0.0 = no filtering)
FACES_ENHANCEMENT_QUALITY_TRIGGER=0.5  # Trigger enhancement below this quality

# Integration
FRAME_SERVER_URL=http://frame-server:5001

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Model Caching
INSIGHTFACE_MODELS_CACHE=/root/.insightface  # Persistent model storage (reduces startup time)

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
  "source": "/media/videos/scene.mp4",
  "source_id": "12345",
  "parameters": {
    "min_confidence": 0.9,
    "sampling_interval": 2.0,
    "enable_deduplication": true,
    "embedding_similarity_threshold": 0.6,
    "scene_boundaries": [
      { "start_timestamp": 0.0, "end_timestamp": 15.0 },
      { "start_timestamp": 15.0, "end_timestamp": 30.0 }
    ]
  }
}
```

### Performance Characteristics

| Configuration   | FPS   | 60s Video | Use Case    |
| --------------- | ----- | --------- | ----------- |
| CPU Mode        | 2.5   | 24s       | Development |
| GPU Mode        | 10-15 | 4-6s      | Production  |
| CPU + Sprites   | 5-10  | 6-12s     | Fallback    |
| GPU + Scene Opt | 15-20 | 3-4s      | Optimized   |

**Memory:**

- CPU: ~600 MB (model + runtime)
- GPU: ~1.5 GB VRAM
- Per face: ~2 KB (512 floats)

**Scaling:**

- Single video: 1 worker
- Multiple videos: N workers, parallel jobs
- High-volume: Load balancer + pool

---

## Occlusion Detection

**Model:** ResNet18 (Custom-trained)
**Training Data:** 30k+ images from MAFA dataset (masks, hands, sunglasses)
**Input:** 224x224 RGB
**Output:** Binary classification (occluded/not occluded) + probability (0.0-1.0)
**Threshold:** 0.5 for binary decision
**Status:** Integrated - November 2025

### Model History

**v2 (Current) - ResNet18:**

- ~100% TPR on hand-occluded faces
- 42.6 MB model size
- ~3.7ms inference time
- Trained on 30k+ MAFA dataset

**v1 (Deprecated) - ConvNeXt-Small:**

- ~50% TPR on hand-occluded faces (FAILED)
- 192 MB model size
- ~11.5ms inference time
- Trained on 9,749 images from LamKser dataset

### Overview

The Faces Service includes automatic occlusion detection for all face detections. Occlusion detection identifies faces partially obscured by:

- Sunglasses or eyeglasses
- Face masks (medical, decorative)
- Hands or other objects
- Hair covering parts of the face
- Other types of facial obstruction

### Detection Output

Each face detection includes nested quality and occlusion objects:

```json
{
  "quality": {
    "composite": 0.78,
    "components": {
      "size": 0.85,
      "pose": 0.92,
      "occlusion": 0.58,
      "sharpness": 0.72
    }
  },
  "occlusion": {
    "occluded": false,
    "probability": 0.42
  }
}
```

- **occlusion.occluded** (boolean): True if probability > 0.5
- **occlusion.probability** (float): Confidence score 0.0-1.0
- **quality.composite** (float): Overall quality (weighted average of components)
- **quality.components** (object): Individual quality factors

### Model Details

**Architecture:** ResNet18

- Parameters: 11.1M
- Inference time: ~3.7ms (CPU), <1ms (GPU)
- Model size: 42.6 MB (ONNX)
- Training dataset: 30k+ images from MAFA (Masked Faces) dataset
- Custom trained with hand-occlusion emphasis

**Preprocessing:**

- Resize face crop to 224x224
- Convert BGR to RGB
- Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Inference:**

- ONNX Runtime with CPUExecutionProvider (GPU support available via INSIGHTFACE_DEVICE=cuda)
- Binary classification via softmax output
- Argmax for class prediction (no manual threshold)

### Building the Occlusion Model

The ResNet18 occlusion model was custom-trained on the MAFA dataset.

#### Training Details

**Dataset:** MAFA (Masked Faces) + augmented hand-occlusion samples

- Total images: 30,000+
- Categories: masks, sunglasses, hands, clean faces
- Augmentation: rotation, flipping, brightness adjustment

**Training Parameters:**

- Architecture: ResNet18 (torchvision)
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 50
- Batch size: 32

**Export to ONNX:**

```python
import torch
import torch.onnx

model = load_trained_model()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "occlusion_classifier.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

#### Verify Output

```bash
file faces-service/models/occlusion_classifier.onnx
# Expected: data (binary ONNX format)

ls -lh faces-service/models/occlusion_classifier.onnx
# Expected: ~42.6 MB
```

#### Integration

The model is automatically loaded at service startup and used for all face detections:

```python
# faces-service/app/occlusion_detector.py
detector = OcclusionDetector()  # Loads ONNX model
occluded, probability = detector.detect(face_crop)
```

### Client Filtering

Occlusion data is returned for all faces - filtering is the responsibility of the client:

```python
# Example: Filter out highly occluded faces
faces = [f for f in detections if f['occlusion_probability'] < 0.7]

# Example: Flag occluded faces for manual review
flagged = [f for f in detections if f['occluded']]
```

### Performance Impact

- **Additional latency per face:** ~3-4ms (CPU), <1ms (GPU)
- **Memory overhead:** ~43 MB (model) + ~1 MB (runtime)
- **Throughput impact:** Minimal (~2-3% slower overall pipeline)
- **Graceful degradation:** If model fails to load, returns `occluded=False, probability=0.0`

### Model Selection History

**Initial Evaluation (LamKser pre-trained models):**

| Model          | Accuracy | Hand TPR | Params | Inference | Notes           |
| -------------- | -------- | -------- | ------ | --------- | --------------- |
| ConvNeXt-Small | 98.87%   | **50%**  | 49.4M  | 11.54ms   | Failed on hands |
| ResNet18       | 97.03%   | **50%**  | 11.1M  | 3.69ms    | Failed on hands |

**Issue:** All LamKser pre-trained models failed to detect hand-on-face occlusions (~50% TPR).

**Solution:** Custom-trained ResNet18 on MAFA dataset with hand-occlusion augmentation.

**Final Model (Current):**

- Architecture: ResNet18 (torchvision)
- Hand TPR: ~100%
- Size: 42.6 MB
- Speed: ~3.7ms (CPU)
- Training data: 30k+ MAFA images

The custom-trained model significantly outperforms pre-trained alternatives on hand-occlusion detection while maintaining excellent performance on masks and sunglasses.

---

**Last Updated:** 2025-11-24
**Status:** Implemented and Tested
