# How to Use Stash Auto Vision

## Quick Start

```bash
# Start services
docker compose up -d

# Check health
curl http://localhost:5010/health

# Submit job
curl -X POST http://localhost:5010/vision/analyze/faces \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/scene_12345.mp4",
    "scene_id": "12345"
  }'

# Check status
curl http://localhost:5010/vision/analyze/faces/jobs/{job_id}/status

# Get results
curl http://localhost:5010/vision/analyze/faces/jobs/{job_id}/results
```

## Face Detection

**Simple:**
```bash
curl -X POST http://localhost:5010/vision/analyze/faces \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/scene.mp4",
    "scene_id": "123"
  }'
```

**With options:**
```json
{
  "video_path": "/media/videos/scene.mp4",
  "scene_id": "123",
  "parameters": {
    "min_confidence": 0.9,
    "max_faces": 50,
    "enable_clustering": true
  }
}
```

## Face Enhancement

**Enable in environment:**
```bash
# Edit .env
ENABLE_ENHANCEMENT=true
ENHANCEMENT_MODEL=gfpgan

# Restart
docker compose restart frame-server
```

**Use in requests:**
```json
{
  "video_path": "/media/videos/scene.mp4",
  "enhancement": {
    "enabled": true,
    "fidelity_weight": 0.7
  }
}
```

**Single frame:**
```bash
curl "http://localhost:5001/extract-frame?video_path=/media/videos/scene.mp4&timestamp=1.0&enhance=1&fidelity_weight=0.7"
```

**Fidelity weight:** 0.0 (preserve original) to 1.0 (max enhancement). Default: 0.7
**Note:** First run downloads models (~500MB). Enhancement upscales 2x, some detail loss possible.

## Scene Detection

```bash
curl -X POST http://localhost:5010/vision/analyze/scenes \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/scene.mp4",
    "scene_id": "123",
    "parameters": {
      "threshold": 27.0
    }
  }'
```

## Combined Analysis

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/media/videos/scene.mp4",
    "scene_id": "123",
    "modules": {
      "scenes": {"enabled": true},
      "faces": {"enabled": true}
    }
  }'
```

## Configuration

**Confidence Thresholds:**
```bash
# Environment defaults (in .env)
FACES_MIN_CONFIDENCE=0.9
SCENES_THRESHOLD=27.0

# Per-request override
{
  "parameters": {
    "face_min_confidence": 0.7  # Lower for difficult videos
  }
}
```

**GPU vs CPU:**
```bash
# GPU mode (.env)
OPENCV_DEVICE=cuda
INSIGHTFACE_DEVICE=cuda

# CPU mode (.env.cpu.example)
OPENCV_DEVICE=cpu
INSIGHTFACE_DEVICE=cpu
```

## Troubleshooting

**Service not starting:**
```bash
docker compose logs {service-name}
```

**Cache issues:**
```bash
docker exec -it vision-redis redis-cli KEYS "*"
docker exec -it vision-redis redis-cli FLUSHALL  # Clear cache
```

**Port conflicts (macOS):**
```bash
# Use port 5010 (5000 conflicts with AirPlay)
VISION_API_PORT=5010
```

**Enhancement not working:**
```bash
# Check if enabled
docker compose logs frame-server | grep -i enhancement

# First run downloads models (~500MB, ~40s on startup)
docker compose logs frame-server | grep -i "Downloading\|GFPGAN initialized"

# Service will fail to start if ENABLE_ENHANCEMENT=true but init fails
```

## API Documentation

Interactive docs: `http://localhost:5010/docs`

Full API spec: See `openapi.yml`
