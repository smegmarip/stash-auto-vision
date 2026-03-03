# Captioning Service

**JoyCaption VLM Integration for Video Captioning**

The Captioning Service provides AI-powered video captioning using JoyCaption Alpha Two, a vision-language model based on Llama 3.1 8B. It generates detailed tags and descriptions for video frames, with optional alignment to Stash's tag taxonomy.

---

## Overview

### Key Features

- **JoyCaption VLM**: Llama 3.1 8B-based vision-language model
- **Multiple Prompt Types**: Booru-like tags, descriptive, straightforward, and more
- **4-bit Quantization**: Reduces VRAM from ~17GB to ~8GB
- **Tag Alignment**: Map free-form VLM output to Stash taxonomy via fuzzy matching
- **Scene-Aware**: Integrates with scenes-service for per-scene captioning
- **GPU Orchestration**: Coordinates with resource-manager for VRAM allocation

### VRAM Requirements

| Mode | VRAM Usage | Notes |
|------|------------|-------|
| Full precision (fp16) | ~17GB | Not recommended for RTX A4000 |
| 4-bit quantized | ~8GB | Recommended for production |

---

## API Endpoints

### Submit Captioning Job

```bash
POST /captions/analyze
```

**Request:**
```json
{
  "source": "/media/videos/scene_001.mp4",
  "source_id": "stash_scene_123",
  "scenes_job_id": "scenes-abc123",
  "parameters": {
    "prompt_type": "booru_like",
    "frame_selection": "scene_based",
    "frames_per_scene": 3,
    "min_confidence": 0.5,
    "align_to_taxonomy": true,
    "use_quantization": true
  }
}
```

**Response:**
```json
{
  "job_id": "captions-550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Captioning job queued",
  "created_at": "2025-12-02T12:34:56.789Z",
  "cache_hit": false
}
```

### Get Job Status

```bash
GET /captions/jobs/{job_id}/status
```

**Response:**
```json
{
  "job_id": "captions-550e8400...",
  "status": "processing",
  "progress": 0.65,
  "stage": "captioning",
  "message": "Captioning frame 13/20",
  "gpu_wait_position": null
}
```

### Get Job Results

```bash
GET /captions/jobs/{job_id}/results
```

**Response:**
```json
{
  "job_id": "captions-550e8400...",
  "source_id": "stash_scene_123",
  "status": "completed",
  "captions": {
    "frames": [
      {
        "frame_index": 0,
        "timestamp": 5.0,
        "raw_caption": "1girl, solo, brown_hair, long_hair, sitting, couch, living_room, casual_clothing",
        "tags": [
          {"tag": "solo", "confidence": 0.95, "source": "aligned_exact", "stash_tag_id": "123"},
          {"tag": "brown_hair", "confidence": 0.90, "source": "aligned_exact", "stash_tag_id": "456"}
        ],
        "scene_index": 0,
        "prompt_type_used": "booru_like"
      }
    ],
    "scene_summaries": [
      {
        "scene_index": 0,
        "start_timestamp": 0.0,
        "end_timestamp": 30.0,
        "dominant_tags": ["solo", "indoor", "conversation"],
        "frame_count": 3,
        "avg_confidence": 0.85
      }
    ]
  },
  "metadata": {
    "frames_captioned": 20,
    "processing_time_seconds": 45.2,
    "vram_peak_mb": 8500,
    "gpu_wait_time_seconds": 12.5
  }
}
```

### Health Check

```bash
GET /captions/health
```

---

## Prompt Types

| Type | Description | Use Case |
|------|-------------|----------|
| `booru_like` | Comma-separated tags, booru style | Primary for tag extraction |
| `straightforward` | Brief, direct description | Fallback for poor results |
| `descriptive` | Formal detailed description | Scene understanding |
| `descriptive_informal` | Casual description | User-facing summaries |
| `art_critic` | Analytical composition review | Artistic content |
| `training_prompt` | Image generation prompt format | Data export |

### Recommended Configuration

For social media content analysis:
- Primary: `booru_like` (comprehensive tagging)
- Fallback: `straightforward` (when tags are too sparse)

---

## Frame Selection Methods

### Scene-Based (Recommended)

Extracts N frames per scene boundary:
- Beginning, middle, and end of each scene
- Requires `scenes_job_id` parameter
- Best for diverse content coverage

### Interval-Based

Extracts frames at fixed intervals:
- Simple but may miss important moments
- Good for consistent content

### Sprite Sheet

Uses pre-generated sprite sheets:
- Fastest option
- Requires frame-server sprite cache

---

## Tag Alignment

The service can align free-form VLM output to your Stash tag taxonomy:

### Alignment Methods

1. **Exact Match**: Direct name match
2. **Alias Match**: Match via tag aliases (95% confidence)
3. **Fuzzy Match**: Similarity-based matching (scaled by similarity score)

### Configuration

```json
{
  "align_to_taxonomy": true,
  "min_confidence": 0.5
}
```

### Sync Taxonomy

```bash
POST /captions/taxonomy/sync
```

Refreshes the tag taxonomy from Stash.

---

## GPU Resource Management

The captioning service coordinates with resource-manager for GPU access:

1. Service requests GPU with estimated VRAM
2. If GPU available, access granted immediately
3. Otherwise, request queued by priority
4. Job status shows `waiting_for_gpu` with queue position
5. Periodic heartbeats keep lease active
6. GPU released automatically on completion

### Job Status During GPU Wait

```json
{
  "status": "waiting_for_gpu",
  "gpu_wait_position": 2,
  "message": "Waiting for GPU access"
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPTION_DEVICE` | `cuda` | Device (cuda/cpu) |
| `USE_QUANTIZATION` | `true` | Enable 4-bit quantization |
| `CAPTIONING_STUB_MODE` | `true` | Stub mode (no model) |
| `CAPTIONS_MIN_CONFIDENCE` | `0.5` | Default min confidence |
| `STASH_URL` | `http://localhost:9999` | Stash instance URL |
| `STASH_API_KEY` | - | Stash API key |
| `RESOURCE_MANAGER_URL` | `http://resource-manager:5007` | Resource manager |

---

## Performance

### Estimated Processing Times

| Content | Frames | Time (quantized) |
|---------|--------|------------------|
| 5 min video | 20 | ~40 seconds |
| 10 min video | 40 | ~80 seconds |
| 30 min video | 100 | ~4 minutes |

### Memory Usage

- Model: ~8GB VRAM (quantized)
- Per-frame: ~500MB additional
- Peak: ~9GB VRAM

---

## Integration with Stash

### Setup

1. Configure Stash URL and API key in environment
2. Create tag taxonomy in Stash (see below)
3. Sync taxonomy: `POST /captions/taxonomy/sync`
4. Submit captioning jobs with `align_to_taxonomy: true`

### Recommended Taxonomy

Create hierarchical tags in Stash:
```
Person
├── Solo
├── Couple
├── Group
└── ...

Action
├── Standing
├── Sitting
├── Walking
└── ...

Setting
├── Indoor
│   ├── Bedroom
│   ├── Living Room
│   └── ...
└── Outdoor
    ├── Beach
    ├── Park
    └── ...
```

---

## Troubleshooting

### Model Not Loading

```bash
# Check VRAM availability
nvidia-smi

# Check logs
docker logs vision-captioning-service
```

### Poor Caption Quality

- Try different prompt types
- Increase frames_per_scene
- Check video quality

### GPU Wait Too Long

- Check resource-manager queue
- Lower priority of background jobs
- Consider reducing concurrent services

---

## Related Documentation

- [Resource Manager](RESOURCE_MANAGER.md)
- [Scenes Service](SCENES_SERVICE.md)
- [Frame Server](FRAME_SERVER.md)
