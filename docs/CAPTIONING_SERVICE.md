# Captioning Service

**JoyCaption VLM Integration for Video Captioning**

The Captioning Service provides AI-powered video captioning using JoyCaption Alpha Two, a vision-language model based on Llama 3.1 8B. It generates detailed tags and structured scene summaries for video frames, with hierarchical tag alignment to Stash's taxonomy.

---

## Overview

### Key Features

- **JoyCaption VLM**: Llama 3.1 8B-based vision-language model with SigLIP vision encoder
- **Structured Scene Summaries**: JSON output with locale, persons, activities, cinematography, mood, and more
- **Hierarchical Tag Scoring**: DFS-based taxonomy traversal with tag description disambiguation
- **Sharpest Frame Selection**: Laplacian variance-based quality filtering
- **4-bit Quantization**: Reduces VRAM from ~17GB to ~8GB
- **Sprite Sheet Support**: Ultra-fast frame extraction from pre-generated sprites
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
    "prompt_type": "scene_summary",
    "frame_selection": "scene_based",
    "frames_per_scene": 3,
    "min_confidence": 0.5,
    "align_to_taxonomy": true,
    "use_hierarchical_scoring": true,
    "select_sharpest": true,
    "sharpness_candidate_multiplier": 3,
    "use_quantization": true,
    "sprite_vtt_url": "http://stash:9999/scene/123/vtt/thumbs.vtt",
    "sprite_image_url": "http://stash:9999/scene/123/sprite"
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
        "raw_caption": "{\"locale\": \"indoor\", \"setting\": \"living room\", ...}",
        "summary": {
          "locale": "indoor",
          "setting": "living room",
          "persons": {
            "count": 1,
            "details": [
              {
                "gender": "female",
                "age_range": "young_adult",
                "hair": "brown, long",
                "expression": "neutral",
                "pose": "sitting"
              }
            ]
          },
          "attire": ["casual dress", "sandals"],
          "objects": ["couch", "coffee table", "lamp"],
          "activities": ["conversation"],
          "cinematography": {
            "shot_type": "medium",
            "camera_angle": "eye_level",
            "framing": "half_body"
          },
          "visual_style": {
            "color_palette": ["warm", "beige", "brown"],
            "color_grading": "warm",
            "quality": "hd"
          },
          "environment": {
            "time_of_day": "afternoon",
            "ambient_light": "bright"
          },
          "mood": "relaxed",
          "genre": "drama"
        },
        "tags": [
          {"tag": "Indoor", "confidence": 0.95, "source": "hierarchical", "stash_tag_id": "123"},
          {"tag": "Living Room", "confidence": 0.90, "source": "hierarchical", "stash_tag_id": "456"}
        ],
        "scene_index": 0,
        "prompt_type_used": "scene_summary",
        "sharpness_score": 0.87
      }
    ],
    "scene_summaries": [
      {
        "scene_index": 0,
        "start_timestamp": 0.0,
        "end_timestamp": 30.0,
        "dominant_tags": ["Indoor", "Living Room", "Conversation"],
        "frame_count": 3,
        "avg_confidence": 0.85
      }
    ]
  },
  "metadata": {
    "frames_captioned": 20,
    "frames_analyzed": 60,
    "sharpness_filtered": true,
    "processing_time_seconds": 45.2,
    "vram_peak_mb": 8500,
    "gpu_wait_time_seconds": 12.5
  }
}
```

### Upload Taxonomy

```bash
POST /captions/taxonomy/upload
```

Upload taxonomy directly without Stash connection:

```json
[
  {
    "id": "1",
    "name": "Indoor",
    "description": "Interior spaces, buildings, enclosed areas",
    "aliases": ["indoors", "inside"],
    "parent_id": null,
    "children": ["2", "3"]
  },
  {
    "id": "2",
    "name": "Living Room",
    "description": "Main room for relaxation and socializing",
    "parent_id": "1",
    "children": []
  }
]
```

### Sync Taxonomy from Stash

```bash
POST /captions/taxonomy/sync
```

Refreshes the tag taxonomy from Stash.

### Health Check

```bash
GET /captions/health
```

---

## Prompt Types

| Type | Description | Use Case |
|------|-------------|----------|
| `scene_summary` | Structured JSON with detailed scene analysis | Primary for comprehensive tagging |
| `booru_like` | Comma-separated tags, booru style | Tag extraction |
| `booru_like_extended` | Extended booru tags with more detail | Comprehensive tagging |
| `straightforward` | Brief, direct description | Fallback for poor results |
| `descriptive` | Formal detailed description | Scene understanding |
| `descriptive_informal` | Casual description | User-facing summaries |
| `art_critic` | Analytical composition review | Artistic content |
| `training_prompt` | Image generation prompt format | Data export |

### Scene Summary Fields

The `scene_summary` prompt type returns structured JSON with:

**Location:**
- `locale`: indoor/outdoor with geographic type
- `setting`: specific environment (bedroom, office, beach)
- `location_details`: additional specifics

**Persons:**
- `persons.count`: number of people
- `persons.details[]`: gender, age_range, ethnicity, body_type, hair, expression, pose, position

**Objects & Elements:**
- `objects[]`: notable props and items
- `furniture[]`: furniture visible
- `attire[]`: clothing items for each person
- `background_elements[]`, `foreground_elements[]`
- `text_visible`: any visible text or signage

**Actions:**
- `activities[]`: actions being performed
- `action_intensity`: static/mild/moderate/intense
- `interactions`: how people interact

**Technical:**
- `cinematography`: shot_type, camera_angle, camera_movement, focus, composition, framing
- `visual_style`: color_palette, color_grading, contrast, saturation, film_grain, quality, era_aesthetic
- `environment`: time_of_day, weather, season, atmosphere, ambient_light
- `lighting`, `lighting_type`

**Mood & Genre:**
- `mood`: emotional tone
- `tension_level`: none/low/medium/high
- `genre`, `sub_genre`, `content_type`
- `narrative_context`: what seems to be happening
- `notable_features[]`: anything unusual

---

## Frame Selection Methods

### Scene-Based (Recommended)

Extracts N frames per scene boundary:
- Beginning, middle, and end of each scene
- Requires `scenes_job_id` parameter
- Best for diverse content coverage

### Sharpest Frame Selection

When `select_sharpest: true`:
1. Extracts `frames_per_scene * sharpness_candidate_multiplier` candidates
2. Calculates Laplacian variance (sharpness) for each
3. Selects sharpest N frames per scene
4. Returns `sharpness_score` in results

### Interval-Based

Extracts frames at fixed intervals:
- Simple but may miss important moments
- Good for consistent content

### Sprite Sheet

Uses pre-generated sprite sheets from Stash:
- Fastest option (100+ FPS)
- Bypasses video decoding entirely
- Requires `sprite_vtt_url` and `sprite_image_url` parameters

---

## Hierarchical Tag Scoring

The service uses DFS pre-order traversal for intelligent tag matching:

### Features

1. **Hierarchical Inheritance**: Parent tag matches boost child tag scores
2. **Tag Descriptions**: Disambiguates similar tags (e.g., "tails" - animal vs coin flip)
3. **Multiple Match Types**: exact, alias, fuzzy, description, hierarchical
4. **Score Decay**: Configurable decay per hierarchy level (default 0.8)

### Tag Description Disambiguation

Tags with descriptions receive context-aware scoring:

```json
{
  "id": "123",
  "name": "tails",
  "description": "animal appendage, not coin flip",
  "aliases": ["tail"]
}
```

When "tails" appears in text:
- If context includes "animal", "fur", "pet" → boosted score
- If context includes "coin", "flip", "heads" → penalized score

### Configuration

```json
{
  "align_to_taxonomy": true,
  "use_hierarchical_scoring": true,
  "min_confidence": 0.3
}
```

---

## GPU Resource Management

The captioning service coordinates with resource-manager for GPU access:

1. Service requests GPU with estimated VRAM (~8GB)
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
| `CAPTIONING_STUB_MODE` | `false` | Stub mode (no model) |
| `CAPTIONS_MIN_CONFIDENCE` | `0.5` | Default min confidence |
| `STASH_URL` | `http://localhost:9999` | Stash instance URL |
| `STASH_API_KEY` | - | Stash API key |
| `RESOURCE_MANAGER_URL` | `http://resource-manager:5007` | Resource manager |

### Request Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt_type` | `scene_summary` | Prompt type for captioning |
| `frame_selection` | `scene_based` | How to select frames |
| `frames_per_scene` | `3` | Frames per scene (1-10) |
| `sampling_interval` | `5.0` | Seconds between frames (interval mode) |
| `min_confidence` | `0.5` | Minimum tag confidence |
| `max_tags_per_frame` | `20` | Maximum tags per frame |
| `align_to_taxonomy` | `true` | Align to Stash tags |
| `use_hierarchical_scoring` | `true` | Use DFS scoring |
| `select_sharpest` | `true` | Filter by sharpness |
| `sharpness_candidate_multiplier` | `3` | Candidates per final frame |
| `batch_size` | `1` | Batch size for inference |
| `use_quantization` | `true` | Use 4-bit quantization |
| `sprite_vtt_url` | - | URL to sprite VTT file |
| `sprite_image_url` | - | URL to sprite grid image |

---

## Performance

### Estimated Processing Times

| Content | Frames | Time (quantized) |
|---------|--------|------------------|
| 5 min video | 20 | ~40 seconds |
| 10 min video | 40 | ~80 seconds |
| 30 min video | 100 | ~4 minutes |

With sharpness filtering (3x candidates):
- Add ~10-20% for sharpness analysis
- Significantly improves frame quality

With sprite sheets:
- Frame extraction: <1 second (vs 5-10 seconds)

### Memory Usage

- Model: ~8GB VRAM (quantized)
- Per-frame: ~500MB additional
- Peak: ~9GB VRAM

---

## Stash Plugin

A Stash plugin is available for automatic scene captioning:

### Installation

1. Copy `stash-plugin/` to Stash plugins directory
2. Enable "Auto Vision Captioning" plugin in Stash settings
3. Configure captioning service URL

### Features

- **Auto-Caption on Scan**: Automatically caption new scenes
- **Manual Captioning**: Caption selected scenes via task
- **Tag Application**: Apply matched tags to scenes
- **Sprite Sheet Support**: Use Stash's sprite sheets for fast extraction

### Configuration

| Setting | Description |
|---------|-------------|
| `api_url` | Captioning service URL |
| `prompt_type` | Prompt type for captioning |
| `min_confidence` | Minimum confidence threshold |
| `auto_caption_enabled` | Enable auto-captioning |
| `apply_tags` | Apply tags to scenes |
| `max_tags_per_scene` | Maximum tags per scene |
| `use_hierarchical_scoring` | Use DFS scoring |
| `use_sprite_sheets` | Use sprite sheets |

---

## Integration with Stash

### Setup

1. Configure Stash URL and API key in environment
2. Create tag taxonomy in Stash with descriptions
3. Sync taxonomy: `POST /captions/taxonomy/sync`
4. Submit captioning jobs with `align_to_taxonomy: true`

### Recommended Taxonomy

Create hierarchical tags in Stash with descriptions:

```
Person
├── Solo (single person in frame)
├── Couple (two people together)
├── Group (three or more people)
└── ...

Action
├── Standing (person standing upright)
├── Sitting (person seated)
├── Walking (person in motion)
└── ...

Setting
├── Indoor (interior spaces)
│   ├── Bedroom (sleeping quarters)
│   ├── Living Room (main relaxation area)
│   └── ...
└── Outdoor (exterior spaces)
    ├── Beach (sandy shore near water)
    ├── Park (public green space)
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

- Use `scene_summary` prompt type for structured output
- Enable `select_sharpest` for better frame quality
- Increase `frames_per_scene` and `sharpness_candidate_multiplier`
- Check video quality and encoding

### Tags Not Aligning

- Verify taxonomy is synced: `GET /captions/taxonomy`
- Add tag descriptions for disambiguation
- Lower `min_confidence` threshold
- Check tag aliases in Stash

### GPU Wait Too Long

- Check resource-manager queue
- Lower priority of background jobs
- Consider reducing concurrent services

---

## Related Documentation

- [Resource Manager](RESOURCE_MANAGER.md)
- [Scenes Service](SCENES_SERVICE.md)
- [Frame Server](FRAME_SERVER.md)
