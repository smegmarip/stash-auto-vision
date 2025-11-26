# Test Datasets Overview

This document describes all datasets available for testing the Stash Auto Vision services.

---

## Available Datasets

### 1. Selfies Dataset (Small, Quick Testing)

**Purpose:** Fast validation of face recognition pipeline

**Location:** `data/selfies/`

**Structure:**

```
selfies/
├── 1/                    # Subject 1
│   ├── 1.jpg            # Photo
│   ├── 2.jpg            # Photo
│   ├── 3.mp4            # Video
│   ├── 4.mp4            # Video
│   └── ...              # 4 photos + 4 videos per subject
├── 2/                    # Subject 2
├── ...                   # Subjects 3-10
└── selfie_and_video.csv  # File manifest
```

**Details:**

- **Subjects:** 10 individuals
- **Files per subject:** 4 photos + 4 videos
- **Total:** 40 photos + 40 videos
- **Use case:** Quick smoke tests, clustering validation

**Testing Scenarios:**

```bash
# Test single subject face recognition
curl -X POST http://localhost:5003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/selfies/1/3.mp4",
    "source_id": "selfie_subject_1",
    "parameters": {
      "min_confidence": 0.9,
      "enable_deduplication": true,
      "embedding_similarity_threshold": 0.6
    }
  }'

# Expected: unique_faces = 1
```

**Validation:**

- Each subject's photos/videos should cluster to 1 unique face
- Cross-subject videos should produce separate face_ids
- Embedding similarity within subject >0.6

---

### 2. YouTube Faces DB (Large, Production-like)

**Purpose:** Comprehensive face recognition validation with ground truth

**Location:** `data/youtube_faces/`

**Structure:**

```
youtube_faces/
├── data/
│   ├── frame_images_DB/
│   │   ├── Aaron_Eckhart/
│   │   │   ├── 0/
│   │   │   │   ├── 0.1.jpg
│   │   │   │   ├── 0.2.jpg
│   │   │   │   └── ...
│   │   │   └── 1/
│   │   ├── Aaron_Guiel/
│   │   └── ...              # 1,595 subjects
│   ├── aligned_images_DB/   # Pre-aligned face crops
│   ├── *.labeled_faces.txt  # Ground truth annotations
│   └── README.txt
└── README.md
```

**Details:**

- **Source:** Tel Aviv University (2011)
- **Subjects:** 1,595 individuals
- **Videos:** 3,425 total
- **Frames:** 48 to 6,070 per video (avg: 181.3)
- **Size:** 24.4 GB
- **Ground truth:** Face bounding boxes in .labeled_faces.txt files

**Testing Scenarios:**

```bash
# Test with celebrity subject (multiple videos)
curl -X POST http://localhost:5003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/youtube_faces/data/frame_images_DB/Aaron_Eckhart/0/*.jpg",
    "source_id": "ytfaces_aaron_eckhart",
    "parameters": {
      "min_confidence": 0.9,
      "enable_deduplication": true
    }
  }'

# Expected: unique_faces should match number of distinct people in video
```

**Validation Against Ground Truth:**

```python
# Parse ground truth labels
with open('data/youtube_faces/data/Aaron_Eckhart.labeled_faces.txt') as f:
    ground_truth = []
    for line in f:
        filename, _, x, y, w, h, _, _ = line.strip().split(',')
        ground_truth.append({
            'filename': filename,
            'bbox_center': (int(x), int(y)),
            'bbox_size': (int(w), int(h))
        })

# Compare with faces-service detections
# - Check detection rate (>95%)
# - Validate bounding box accuracy (IoU >0.7)
# - Confirm embedding consistency across same person's videos
```

**Use Cases:**

- Detection accuracy benchmarking
- Embedding quality validation
- Cross-video clustering (same person, different videos)
- Pose variation testing (head rotation angles available)

---

### 3. Charades Dataset (Scene Analysis)

**Purpose:** Scene detection and future semantic analysis

**Location:** `data/charades/`

**Structure:**

```
charades/
├── dataset/
│   ├── 001YG.mp4
│   ├── 002YG.mp4
│   └── ...              # ~9,500 videos
└── annotation/
    ├── Charades_v1_train.csv
    ├── Charades_v1_test.csv
    └── ...
```

**Details:**

- **Source:** AI2 (Allen Institute for AI)
- **Videos:** ~9,500 clips
- **Actions:** 157 action classes
- **Duration:** 30 seconds average per video
- **Use case:** Frame extraction, scene detection, future semantics/objects

**Testing Scenarios:**

```bash
# Test frame extraction performance
curl -X POST http://localhost:5001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/001YG.mp4",
    "extraction_method": "opencv_cuda",
    "sampling_strategy": {
      "mode": "interval",
      "interval_seconds": 2.0
    }
  }'

# Test scene detection
curl -X POST http://localhost:5002/detect \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/001YG.mp4",
    "detection_method": "content",
    "threshold": 27.0
  }'
```

**Current Use (Phase 1):**

- Frame extraction speed benchmarks
- Scene boundary detection accuracy
- Cache performance testing

**Future Use (Phase 2-3):**

- Semantic scene classification (indoor/outdoor, locations)
- Object detection validation
- Action recognition

---

## Dataset Comparison

| Dataset           | Subjects | Videos | Best For                          | Complexity |
| ----------------- | -------- | ------ | --------------------------------- | ---------- |
| **Selfies**       | 10       | 40     | Quick tests, smoke testing        | Low        |
| **YouTube Faces** | 1,595    | 3,425  | Production validation, benchmarks | High       |
| **Charades**      | N/A      | 9,500  | Scene detection, frame extraction | Medium     |

---

## Recommended Testing Workflow

### 1. Development Phase (Quick Validation)

**Use Selfies dataset:**

```bash
# Test all 10 subjects
for i in {1..10}; do
  curl -X POST http://localhost:5003/analyze \
    -H "Content-Type: application/json" \
    -d "{
      \"video_path\": \"/media/selfies/$i/3.mp4\",
      \"source_id\": \"selfie_subject_$i\"
    }"
done

# Validate: Each should produce 1 unique face
# Processing time: <1 minute total
```

### 2. Integration Testing

**Use YouTube Faces (small subset):**

```bash
# Test single subject with multiple videos
# Aaron_Eckhart has multiple videos - should cluster to same person

# Process video 0
curl -X POST http://localhost:5003/analyze \
  -d '{"video_path": "/media/youtube_faces/data/frame_images_DB/Aaron_Eckhart/0/..."}'

# Process video 1
curl -X POST http://localhost:5003/analyze \
  -d '{"video_path": "/media/youtube_faces/data/frame_images_DB/Aaron_Eckhart/1/..."}'

# Validate: Embeddings from both videos should be similar (>0.6)
```

### 3. Production Validation

**Use YouTube Faces (full dataset):**

- Run comprehensive benchmarks
- Compare detection rates against ground truth
- Measure embedding consistency
- Validate clustering accuracy

---

## Docker Volume Configuration

Update `.env` to include all datasets:

```bash
# Charades (scene detection)
SERVER_MEDIA_PATH=/Users/x/dev/resources/repo/stash-auto-vision/data/charades/dataset

# Face recognition datasets
SELFIES_PATH=/Users/x/dev/resources/repo/stash-auto-vision/data/selfies
YOUTUBE_FACES_PATH=/Users/x/dev/resources/repo/stash-auto-vision/data/youtube_faces/data/frame_images_DB
```

Update `docker-compose.yml` for faces-service:

```yaml
faces-service:
  volumes:
    - ${SERVER_MEDIA_PATH}:/media/videos:ro
    - ${SELFIES_PATH}:/media/selfies:ro
    - ${YOUTUBE_FACES_PATH}:/media/youtube_faces:ro
```

---

## Ground Truth Validation

### YouTube Faces - Face Detection Accuracy

```python
# validation_script.py
import json

def calculate_detection_accuracy(ground_truth_file, results_json):
    """
    Compare faces-service results against YouTube Faces ground truth

    Args:
        ground_truth_file: Path to .labeled_faces.txt
        results_json: faces-service API response

    Returns:
        Dict with accuracy metrics
    """
    # Parse ground truth
    gt_detections = parse_labeled_faces(ground_truth_file)

    # Parse faces-service results
    api_detections = results_json['faces']

    # Calculate metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_frame in gt_detections:
        matched = False
        for detection in api_detections:
            if frame_matches(gt_frame, detection):
                if iou(gt_frame['bbox'], detection['bbox']) > 0.7:
                    true_positives += 1
                    matched = True
                    break

        if not matched:
            false_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detection_rate': recall
    }
```

**Target Metrics:**

- **Detection Rate (Recall):** >95%
- **Precision:** >90%
- **Bounding Box IoU:** >0.7
- **F1 Score:** >0.92

---

## Performance Benchmarks by Dataset

### Selfies (Small, Fast)

| Metric                    | GPU Mode | CPU Mode |
| ------------------------- | -------- | -------- |
| Processing time (1 video) | <5s      | <30s     |
| Detection rate            | >98%     | >98%     |
| Clustering accuracy       | >95%     | >95%     |

### YouTube Faces (Large, Comprehensive)

| Metric                                  | GPU Mode | CPU Mode |
| --------------------------------------- | -------- | -------- |
| Processing time (avg video, 181 frames) | <30s     | <5 min   |
| Detection rate vs ground truth          | >95%     | >95%     |
| Embedding consistency (same person)     | >0.8     | >0.8     |
| Cross-video clustering                  | >90%     | >90%     |

### Charades (Scene Detection)

| Metric                       | GPU Mode    | CPU Mode    |
| ---------------------------- | ----------- | ----------- |
| Frame extraction (30s video) | <5s         | <15s        |
| Scene detection speed        | 300-800 FPS | 100-500 FPS |
| Scene boundary accuracy      | 85-90%      | 85-90%      |

---

## Testing Priorities

### Phase 1 (Current) - Face Recognition

1. **Selfies** - Quick smoke tests ✅
2. **YouTube Faces** - Production validation ✅
3. **Charades** - Frame/scene testing ✅

### Phase 2 (Future) - Semantic Analysis

1. **Charades** - Action classification
2. **Custom datasets** - Scene understanding

### Phase 3 (Future) - Object Detection

1. **Charades** - Object detection
2. **COCO** - General objects (future)

---

## Dataset Citations

### Selfies

- **Type:** Custom test dataset
- **License:** Internal testing only

### YouTube Faces

- **Citation:** Lior Wolf, Tal Hassner and Itay Maoz, "Face Recognition in Unconstrained Videos with Matched Background Similarity", CVPR 2011
- **License:** Research use only
- **URL:** http://www.cs.tau.ac.il/~wolf/ytfaces/

### Charades

- **Citation:** Gunnar A. Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev and Abhinav Gupta, "Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding", ECCV 2016
- **License:** Research use only
- **URL:** https://prior.allenai.org/projects/charades

---

**Last Updated:** 2025-11-08
