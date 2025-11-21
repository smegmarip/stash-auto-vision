# Frame Server Documentation

**Service:** Frame Server
**Port:** 5001
**Status:** Phase 1 - Implemented
**Version:** 1.0.0

---

## Summary

The Frame Server provides GPU-accelerated frame extraction from video files. It supports multiple extraction methods (OpenCV CUDA, OpenCV CPU, FFmpeg, and WebVTT sprite sheets) and implements intelligent caching to avoid redundant processing.

### Key Features

- **GPU-Accelerated Extraction:** OpenCV with CUDA support for high-performance frame decoding (200-400 FPS)
- **Multiple Methods:** Supports opencv_cuda, opencv_cpu, ffmpeg, and sprite sheet parsing
- **Smart Caching:** Content-based cache keys with automatic invalidation on file changes
- **Async Processing:** Background job processing with polling API
- **Flexible Sampling:** Interval-based, timestamp-based, and scene-based sampling strategies
- **TTL-Based Cleanup:** Automatic frame cleanup via cron job (2-hour TTL)

### Architecture

The Frame Server uses an async extract-once + TTL cleanup architecture:

1. **Extract frames asynchronously** (background job)
2. **Serve frames on-demand** (polling with wait support)
3. **Auto-cleanup after TTL** (cron-based garbage collection - 2 hours)
4. **Process while extracting** (async streaming)

---

## OpenAPI 3.0 Schema

```yaml
openapi: 3.0.3
info:
  title: Frame Server API
  description: GPU-accelerated frame extraction service
  version: 1.0.0
servers:
  - url: http://frame-server:5001
    description: Docker network
  - url: http://localhost:5001
    description: External access

paths:
  /extract:
    post:
      summary: Submit frame extraction job
      description: Asynchronously extract frames from video file
      operationId: extractFrames
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExtractFramesRequest'
      responses:
        '202':
          description: Job submitted successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExtractJobResponse'
        '400':
          description: Invalid request parameters
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '404':
          description: Video file not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /jobs/{job_id}/status:
    get:
      summary: Get extraction job status
      operationId: getJobStatus
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Job status retrieved
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExtractJobStatus'
        '404':
          description: Job not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /jobs/{job_id}/results:
    get:
      summary: Get extraction results
      operationId: getJobResults
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Extraction results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExtractJobResults'
        '404':
          description: Job not found
        '409':
          description: Job not completed yet

  /frames/{job_id}/{frame_index}:
    get:
      summary: Get specific frame (on-demand)
      description: Retrieve frame with polling support
      operationId: getFrame
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
        - name: frame_index
          in: path
          required: true
          schema:
            type: integer
        - name: wait
          in: query
          schema:
            type: boolean
            default: true
          description: Wait up to 30s for frame to be ready
      responses:
        '200':
          description: Frame image
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        '202':
          description: Extraction in progress (if wait=false)
        '408':
          description: Frame not ready after timeout
        '404':
          description: Job or frame not found

  /extract-frame:
    get:
      summary: Extract single frame
      description: Extract one frame without job tracking (for thumbnails, optionally with face enhancement)
      operationId: extractSingleFrame
      parameters:
        - name: video_path
          in: query
          required: true
          schema:
            type: string
        - name: timestamp
          in: query
          required: true
          schema:
            type: number
            format: float
        - name: output_format
          in: query
          schema:
            type: string
            enum: [jpeg, png]
            default: jpeg
        - name: quality
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 95
        - name: enhance
          in: query
          schema:
            type: boolean
            default: false
          description: Enable face enhancement
        - name: model
          in: query
          schema:
            type: string
            enum: [gfpgan, codeformer]
            default: gfpgan
          description: Face enhancement model to use
        - name: fidelity_weight
          in: query
          schema:
            type: number
            format: float
            minimum: 0.0
            maximum: 1.0
            default: 0.7
          description: Fidelity vs quality tradeoff (higher = more original details)
        - name: upscale
          in: query
          schema:
            type: integer
            enum: [1, 2, 4]
            default: 2
          description: Upscaling factor for enhancement
      responses:
        '200':
          description: Frame image (enhanced if requested)
          headers:
            X-Cache-Hit:
              schema:
                type: string
                enum: [true, false]
              description: Indicates if frame was served from cache
            X-Timestamp:
              schema:
                type: string
              description: Frame timestamp in seconds
            X-Frame-Number:
              schema:
                type: string
              description: Frame number in video
            X-Resolution:
              schema:
                type: string
              description: Frame resolution (e.g., "1920x1080")
            X-Faces-Enhanced:
              schema:
                type: string
              description: Number of faces enhanced (only if enhance=1)
          content:
            image/jpeg:
              schema:
                type: string
                format: binary
        '400':
          description: Invalid parameters
        '404':
          description: Video not found

  /health:
    get:
      summary: Service health check
      operationId: healthCheck
      responses:
        '200':
          description: Service healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

components:
  schemas:
    ExtractFramesRequest:
      type: object
      required:
        - video_path
      properties:
        video_path:
          type: string
          description: Absolute path to video file
          example: "/media/videos/scene.mp4"
        job_id:
          type: string
          description: Optional custom job ID
        extraction_method:
          type: string
          enum: [opencv_cuda, opencv_cpu, ffmpeg, sprites]
          default: opencv_cuda
        sampling_strategy:
          $ref: '#/components/schemas/SamplingStrategy'
        output_format:
          type: string
          enum: [jpeg, png]
          default: jpeg
        quality:
          type: integer
          minimum: 1
          maximum: 100
          default: 95
        cache_duration:
          type: integer
          description: Cache TTL in seconds
          default: 7200
        sprite_vtt_url:
          type: string
          description: URL to WebVTT sprite coordinates (if using sprites)
        sprite_image_url:
          type: string
          description: URL to sprite JPEG grid (if using sprites)

    SamplingStrategy:
      type: object
      required:
        - mode
      properties:
        mode:
          type: string
          enum: [interval, timestamps, scenes]
        interval_seconds:
          type: number
          format: float
          description: Extract every N seconds (for interval mode)
          example: 2.0
        timestamps:
          type: array
          items:
            type: number
            format: float
          description: Specific timestamps to extract (for timestamps mode)
        scene_boundaries:
          type: array
          items:
            $ref: '#/components/schemas/SceneBoundary'
          description: Scene boundaries for scene-based sampling

    SceneBoundary:
      type: object
      properties:
        start_timestamp:
          type: number
          format: float
        end_timestamp:
          type: number
          format: float
        duration:
          type: number
          format: float

    ExtractJobResponse:
      type: object
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [queued, processing, completed, failed]
        created_at:
          type: string
          format: date-time
        cache_key:
          type: string
        estimated_frames:
          type: integer

    ExtractJobStatus:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [queued, processing, completed, failed]
        progress:
          type: number
          format: float
          minimum: 0
          maximum: 1
        message:
          type: string
        error:
          type: string
        created_at:
          type: string
          format: date-time
        started_at:
          type: string
          format: date-time
        completed_at:
          type: string
          format: date-time
        failed_at:
          type: string
          format: date-time

    ExtractJobResults:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
        frames:
          type: array
          items:
            $ref: '#/components/schemas/FrameMetadata'
        video_metadata:
          $ref: '#/components/schemas/VideoMetadata'
        cache_key:
          type: string

    FrameMetadata:
      type: object
      properties:
        frame_index:
          type: integer
        timestamp:
          type: number
          format: float
        url:
          type: string
          description: file:// URL to frame on shared volume
        width:
          type: integer
        height:
          type: integer

    VideoMetadata:
      type: object
      properties:
        duration:
          type: number
          format: float
        fps:
          type: number
          format: float
        total_frames:
          type: integer
        width:
          type: integer
        height:
          type: integer
        codec:
          type: string

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        service:
          type: string
        version:
          type: string
        gpu_available:
          type: boolean
        extraction_method:
          type: string

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

### Frame Extraction Methods

**Per-Frame Fallback:** By default, the frame server uses automatic fallback. If one method fails for a specific frame, it tries the next method in the chain.

**Fallback Chain:** `opencv_cuda/opencv_cpu` → `pyav_hw` → `pyav_sw` → `ffmpeg`

Configure with `ENABLE_FALLBACK=true` (default).

#### 1. opencv_cuda / opencv_cpu

OpenCV with optional CUDA acceleration.

**Performance:** GPU: 200-400 FPS, CPU: 30-60 FPS
**Use Case:** Primary extraction method, fast but may fail on corrupted videos

#### 2. pyav_hw (PyAV Hardware)

FFmpeg via PyAV with hardware acceleration.

**Performance:** ~100-200 FPS
**Use Case:** Automatic fallback, robust for damaged videos

#### 3. pyav_sw (PyAV Software)

FFmpeg via PyAV with software decoding.

**Performance:** ~30-60 FPS
**Use Case:** Maximum compatibility for problematic encodes

#### 4. ffmpeg (CLI)

External FFmpeg subprocess.

**Performance:** 10-30 FPS
**Use Case:** Last-resort fallback

#### 5. sprites (Sprite Sheets)

Parses WebVTT + JPEG grid for ultra-fast access.

**Performance:** 100+ FPS
**Memory:** Minimal
**Limitation:** Fixed frame count, poor coverage for long videos

### Sampling Strategies

#### Interval Mode

Extract frames at regular intervals:

```json
{
  "mode": "interval",
  "interval_seconds": 2.0
}
```

Extracts frames every 2 seconds throughout the video.

#### Timestamps Mode

Extract specific frames:

```json
{
  "mode": "timestamps",
  "timestamps": [0.0, 5.5, 10.2, 15.0]
}
```

Extracts frames at exact timestamps.

#### Scene-Based Mode

Extract representative frames per scene:

```json
{
  "mode": "scenes",
  "scene_boundaries": [
    {"start_timestamp": 0.0, "end_timestamp": 15.0},
    {"start_timestamp": 15.0, "end_timestamp": 30.0}
  ]
}
```

Extracts 3 frames per scene (start, middle, end).

### Face Enhancement

Frame Server includes optional face enhancement using AI models for upscaling and quality improvement.

#### Available Models

**1. GFPGAN (GAN-based)**
- **Method:** Generative Adversarial Network approach
- **Speed:** ~5-10ms per face (CPU mode)
- **Quality:** Good general-purpose enhancement, may over-smooth
- **Use Case:** Quick enhancement for less critical applications

**2. CodeFormer (Transformer-based)** ⭐ **Recommended**
- **Method:** VQ codebook with transformer, [-1,1] normalized tensors
- **Speed:** ~10-15ms per face (CPU mode)
- **Quality:** **Production-grade, comparable to commercial solutions (e.g., Nero)**
- **Tuning Guide:** Lower fidelity (0.1-0.3) for heavy enhancement, higher (0.5-0.7) for detail preservation
- **Use Case:** **Primary recommendation for all face enhancement tasks**

#### Enhancement Parameters

**fidelity_weight** (0.0 - 1.0, default: 0.5)
- Controls tradeoff between enhancement and fidelity to original
- **Lower values (0.1-0.3):** Maximum enhancement, smoothest results (best for low-quality sources)
- **Medium values (0.4-0.6):** Balanced enhancement (general-purpose, HuggingFace default: 0.5)
- **Higher values (0.7-1.0):** Maximum fidelity, minimal enhancement (best for high-quality sources)

**upscale** (1, 2, or 4, default: 2)
- Upscaling factor for output resolution
- 1x: No upscaling (enhancement only)
- 2x: 640×480 → 1280×960
- 4x: 640×480 → 2560×1920

**model** (gfpgan or codeformer, default: codeformer)
- **CodeFormer:** Production-grade quality, comparable to commercial solutions
- **GFPGAN:** Legacy option, may over-smooth details

#### Usage Examples

**Recommended: CodeFormer with balanced settings:**
```bash
curl "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.5"
```

**Low-quality source (heavy enhancement):**
```bash
curl "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.25"
```

**High-quality source (detail preservation):**
```bash
curl "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.7"
```

**4x upscaling for thumbnail:**
```bash
curl "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&upscale=4"
```

#### Implementation Details

**Architecture:**
- Modular design with `BaseEnhancer` abstract class
- Separate modules: `gfpgan_enhancer.py`, `codeformer_enhancer.py`
- Factory pattern in `face_enhancer.py`

**Dependencies:**
- GFPGAN: `gfpgan==1.3.8`, `realesrgan==0.3.0`
- CodeFormer: `codeformer-pip==0.0.4`, `lpips`, `gdown`
- Shared: `torch==2.0.1`, `torchvision==0.15.2`, `basicsr==1.4.2`, `facexlib==0.3.0`

**Model Storage:**
- Models auto-download to persistent Docker volumes (one-time download)
- `enhancement_models` volume: GFPGAN and RealESRGAN models (423 MB)
  - `codeformer.pth` (~359 MB)
  - `RealESRGAN_x2plus.pth` (~64 MB)
- `codeformer_weights` volume: Detection and parsing models (186 MB)
  - `detection_Resnet50_Final.pth` (~104 MB)
  - `parsing_parsenet.pth` (~81 MB)
- `torch_cache` volume: PyTorch model cache
- Total: ~609 MB cached across restarts

### Sprite Sheet Processing

Frame Server can parse WebVTT sprite coordinates and extract tiles from grid images.

**VTT Format:**
```
WEBVTT

00:00:01.000 --> 00:00:02.000
sprite.jpg#xywh=0,0,160,90

00:00:03.000 --> 00:00:04.000
sprite.jpg#xywh=160,0,160,90
```

**Processing:**
1. Download sprite JPEG grid
2. Parse VTT coordinates using regex: `#xywh=(\d+),(\d+),(\d+),(\d+)`
3. Extract tiles: `tile = grid[y:y+h, x:x+w]`
4. Save tiles as individual frames

### Frame Storage and Cleanup

**Storage Location:** `/tmp/frames/{job_id}/frame_000000.jpg`

**Cleanup Strategy:**
- **TTL:** 2 hours (configurable via `FRAME_TTL_HOURS`)
- **Method:** Cron job runs hourly: `/usr/local/bin/cleanup_frames.sh`
- **Logs:** `/var/log/frame-cleanup.log`

**Manual Cleanup:**
```bash
# Inside container
docker exec vision-frame-server /usr/local/bin/cleanup_frames.sh

# Dry run
docker exec -e DRY_RUN=true vision-frame-server /usr/local/bin/cleanup_frames.sh

# Custom TTL
docker exec -e FRAME_TTL_HOURS=1 vision-frame-server /usr/local/bin/cleanup_frames.sh
```

### Cache Strategy

Frame Server implements two-tier caching: Redis for job metadata and file-based for extracted frames.

#### Redis Cache (Job Metadata)

**Cache Key Generation:**
```python
import hashlib
import os
import json

video_mtime = os.path.getmtime(video_path)
params_str = json.dumps(params, sort_keys=True)
cache_str = f"{video_path}:{video_mtime}:frame:{params_str}"
cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
```

**Redis Structure:**
```
frame:job:{job_id}:metadata       # Job metadata
frame:job:{job_id}:frame:{index}  # Frame file path
frame:cache:{cache_key}           # Cache key → job_id mapping
```

**Automatic Invalidation:**
- File modification changes `mtime`
- New `mtime` generates new cache key
- Old cache entries expire after TTL

#### File-Based Cache (Enhanced Frames)

The `/extract-frame` endpoint implements file-based caching to avoid redundant enhancement operations:

**Cache Key Generation:**
```python
cache_params = {
    "timestamp": timestamp,
    "output_format": output_format,
    "quality": quality,
    "enhance": enhance,
    "model": model if enhance else None,
    "fidelity_weight": fidelity_weight if enhance else None
}
cache_key = cache_manager.generate_cache_key(video_path, cache_params)
```

**Cache Storage:**
- **Location:** `/tmp/frames/` (mounted to `frames_cache` Docker volume)
- **Separate paths:** `enhanced_{cache_key}.jpeg` vs `frame_{cache_key}.jpeg`
- **Parameter-specific:** Different cache keys for different enhancement settings

**Cache Behavior:**
- Check cache before extraction (instant return on hit)
- Store extracted frame after processing
- `X-Cache-Hit` header indicates cache status (`true` or `false`)
- Eliminates redundant 60+ second enhancement operations

**Example:**
```bash
# First request: extract + enhance (60 seconds)
curl -i "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.5"
# Response header: X-Cache-Hit: false

# Second request: cached (instant)
curl -i "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.5"
# Response header: X-Cache-Hit: true

# Different parameters: new cache entry (60 seconds)
curl -i "http://localhost:5001/extract-frame?video_path=/media/video.mp4&timestamp=5.0&enhance=1&model=codeformer&fidelity_weight=0.7"
# Response header: X-Cache-Hit: false
```

### Configuration

Environment variables:

```bash
# Frame extraction
EXTRACTION_METHOD=opencv_cuda    # opencv_cuda, opencv_cpu, pyav_hw, pyav_sw, ffmpeg
OPENCV_DEVICE=cuda               # cuda or cpu
ENABLE_FALLBACK=true             # Per-frame fallback chain (default: true)

# Face enhancement
ENABLE_ENHANCEMENT=true          # Enable face enhancement endpoints (default: true)

# Storage
FRAME_DIR=/tmp/frames
FRAME_TTL_HOURS=2               # Cleanup TTL

# Cache
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600                  # Redis TTL (1 hour)

# Logging
LOG_LEVEL=INFO
```

### Performance Characteristics

| Method | GPU Mode | CPU Mode | Use Case |
|--------|----------|----------|----------|
| opencv_cuda | 200-400 FPS | N/A | Primary (fast, fragile) |
| opencv_cpu | N/A | 30-60 FPS | Primary (fast, fragile) |
| pyav_hw | 100-200 FPS | 50-100 FPS | Fallback (robust + fast) |
| pyav_sw | 30-60 FPS | 30-60 FPS | Fallback (robust) |
| ffmpeg | 50-100 FPS | 15-20 FPS | Last resort |
| sprites | 100+ FPS | 100+ FPS | Ultra-fast (pre-extracted) |

**Disk Usage Example (10-min video @ 2fps):**
- Frames: 1200 × 100KB = ~120MB
- TTL: 2 hours
- Max concurrent jobs: ~60
- Max disk usage: ~7.2GB

---

**Last Updated:** 2025-11-21
**Status:** Implemented and Tested (with PyAV fallback, Face Enhancement, and Enhanced Frame Caching)
