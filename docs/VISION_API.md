# Vision API Documentation

**Service:** Vision API (Orchestrator)
**Port:** 5010
**Path:** `/vision`
**Status:** Phase 3 Complete - Orchestrator for scenes, faces, and semantics
**Version:** 3.0.0

---

## Summary

The Vision API is the primary orchestrator and rollup service for the Stash Auto Vision platform. It coordinates the backend analysis modules (scenes, faces, semantics, objects) and provides a unified API for multi-module video analysis. Clients submit a single job describing which modules to run; the Vision API dispatches per-module sub-jobs, tracks aggregate progress in Redis, and returns the combined results.

As an orchestrator, the Vision API does not perform video processing directly. Instead, it calls each enabled service, polls for completion, aggregates results, and manages job state under the `vision:` Redis namespace. Scene boundaries produced by scenes-service are forwarded to downstream modules (faces, semantics) to avoid redundant decoding work.

Processing is sequential by default to avoid GPU memory contention (JoyCaption beta-one alone peaks around 8 GB VRAM during semantic captioning). Parallel execution is reserved for a future phase.

**Downstream services called by Vision API:**

| Name | URL (default) | Endpoint | Notes |
|---|---|---|---|
| scenes | `http://scenes-service:5002` | `/scenes/detect` | Scene boundary detection |
| faces | `http://faces-service:5003` | `/faces/analyze` | InsightFace buffalo_l |
| semantics | `http://semantics-service:5004` | `/semantics/analyze` | Trained multi-view bi-encoder tag classifier + JoyCaption captioning |
| objects | `http://objects-service:5005` | `/objects/analyze` | Stub (Phase 4) |

> **Live schema:** the authoritative, always-current schema is exposed via the schema-service at **<http://localhost:5009/docs>** (Swagger UI) and **<http://localhost:5009/openapi.json>** (raw). The hand-authored OpenAPI block below is a reference snapshot, not the source of truth.

---

## OpenAPI Reference

This block is extracted from the live `openapi.yml` artifact and kept in sync with the FastAPI models in `vision-api/app/models.py`. Treat it as a user-facing reference; for a byte-accurate schema always consult `/openapi.json`.

```yaml
openapi: 3.1.1
info:
  title: Vision API - Orchestrator
  description: Orchestrator for video vision analysis services
  version: 1.0.0

paths:
  /vision/analyze:
    post:
      summary: Submit comprehensive video analysis job
      description: |
        Submit a video for multi-module analysis. Enabled modules are processed
        sequentially. Scene boundaries from scenes-service are forwarded to
        downstream modules (faces, semantics) for optimized sampling.
      operationId: analyzeVideo
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/AnalyzeVideoRequest"
      responses:
        "202":
          description: Analysis job accepted and queued
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeJobResponse"
        "404":
          description: Video file not found (for local file sources)
        "422":
          description: Validation error

  /vision/jobs/{job_id}/status:
    get:
      summary: Get analysis job status
      description: |
        Poll orchestrator job status. Looks up the job across all service
        namespaces (vision, faces, scenes, semantics, objects).
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Job status retrieved
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeJobStatus"
        "404":
          description: Job not found

  /vision/jobs/{job_id}/results:
    get:
      summary: Get aggregated analysis results
      description: |
        Retrieve aggregated results from all enabled modules. Only available
        when job status is "completed".
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Analysis results
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/AnalyzeJobResults"
        "404":
          description: Job not found
        "409":
          description: Job not completed yet

  /vision/jobs:
    get:
      summary: List jobs across all services
      description: |
        List jobs from vision, faces, scenes, semantics, and objects
        namespaces with optional filtering and pagination. Deduplicated
        by cache_key, preferring vision-level jobs.
      parameters:
        - { name: status,          in: query, schema: { type: string, enum: [queued, processing, completed, failed] } }
        - { name: service,         in: query, schema: { type: string, enum: [vision, faces, scenes, semantics, objects] } }
        - { name: source_id,       in: query, schema: { type: string } }
        - { name: source,          in: query, schema: { type: string } }
        - { name: start_date,      in: query, schema: { type: string, format: date-time } }
        - { name: end_date,        in: query, schema: { type: string, format: date-time } }
        - { name: include_results, in: query, schema: { type: boolean, default: false } }
        - { name: limit,           in: query, schema: { type: integer, default: 50, minimum: 1, maximum: 500 } }
        - { name: offset,          in: query, schema: { type: integer, default: 0, minimum: 0 } }
      responses:
        "200":
          description: List of jobs
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/ListJobsResponse"

  /vision/jobs/count:
    get:
      summary: Count jobs matching filters
      description: |
        Efficient count-only variant of /vision/jobs for pagination and
        tab displays. Returns a total plus a per-service breakdown.
      responses:
        "200":
          description: Count result
          content:
            application/json:
              schema:
                type: object
                properties:
                  total:
                    type: integer
                  by_service:
                    type: object
                    additionalProperties:
                      type: integer

  /vision/health:
    get:
      summary: Service health check
      description: |
        Returns Vision API health along with aggregated health from
        scenes, faces, semantics, and objects services.
      responses:
        "200":
          description: Service healthy
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/HealthResponse"

components:
  schemas:
    AnalyzeVideoRequest:
      type: object
      required: [source, source_id]
      properties:
        source:
          type: string
          description: Path, URL, or image source to analyze
          example: /media/videos/scene.mp4
        source_id:
          type: string
          description: Scene ID for reference (Stash scene ID)
          example: "12345"
        job_id:
          type: string
          description: Optional custom job ID (generated if not provided)
        processing_mode:
          type: string
          enum: [sequential, parallel]
          default: sequential
          description: |
            Sequential processes one module at a time (default, resource-efficient).
            Parallel is reserved for a future phase.
        modules:
          $ref: "#/components/schemas/ModulesConfig"

    ModulesConfig:
      type: object
      properties:
        scenes:    { $ref: "#/components/schemas/ModuleConfig" }   # default enabled: true
        faces:     { $ref: "#/components/schemas/ModuleConfig" }   # default enabled: true
        semantics: { $ref: "#/components/schemas/ModuleConfig" }   # default enabled: false
        objects:   { $ref: "#/components/schemas/ModuleConfig" }   # default enabled: false

    ModuleConfig:
      type: object
      properties:
        enabled:
          type: boolean
          default: true
        parameters:
          type: object
          description: |
            Module-specific parameters. See each service's documentation
            for accepted keys:
              - scenes:    docs/SCENES_SERVICE.md
              - faces:     docs/FACES_SERVICE.md
              - semantics: docs/SEMANTICS_SERVICE.md
              - objects:   docs/OBJECTS_SERVICE.md

    AnalyzeJobResponse:
      type: object
      required: [job_id, status, created_at]
      properties:
        job_id:         { type: string }
        status:         { type: string, enum: [queued, processing, completed, failed] }
        created_at:     { type: string, format: date-time }
        services_enabled:
          type: object
          properties:
            scenes:    { type: boolean }
            faces:     { type: boolean }
            semantics: { type: boolean }
            objects:   { type: boolean }
        processing_mode:
          type: string
          enum: [sequential, parallel]

    AnalyzeJobStatus:
      type: object
      required: [job_id, status, progress, created_at]
      properties:
        job_id:          { type: string }
        status:          { type: string, enum: [queued, processing, completed, failed] }
        progress:        { type: number, minimum: 0, maximum: 1 }
        processing_mode: { type: string, enum: [sequential, parallel] }
        stage:           { type: string, nullable: true, description: "Current processing stage (e.g. scene_detection, face_recognition, semantic_analysis)" }
        message:         { type: string, nullable: true }
        services:
          type: array
          description: Individual service sub-statuses (currently returned empty; reserved for future use)
          items:
            type: object
        created_at:    { type: string, format: date-time }
        started_at:    { type: string, format: date-time, nullable: true }
        completed_at:  { type: string, format: date-time, nullable: true }
        result_summary:{ type: object, nullable: true }
        error:         { type: string, nullable: true }

    AnalyzeJobResults:
      type: object
      required: [job_id, source_id, status]
      properties:
        job_id:    { type: string }
        source_id: { type: string }
        status:    { type: string, enum: [completed] }
        scenes:    { type: object, nullable: true, description: "scenes-service rollup" }
        faces:     { type: object, nullable: true, description: "faces-service rollup" }
        semantics: { type: object, nullable: true, description: "semantics-service rollup (tags, frame_captions, scene_summary, scene_embedding)" }
        objects:   { type: object, nullable: true, description: "objects-service rollup or stub response" }
        metadata:
          type: object
          properties:
            processing_time_seconds: { type: number }
            processing_mode:         { type: string, enum: [sequential, parallel] }
            services_used:
              type: object
              properties:
                scenes:    { type: boolean }
                faces:     { type: boolean }
                semantics: { type: boolean }
                objects:   { type: boolean }

    HealthResponse:
      type: object
      required: [status, service]
      properties:
        status:  { type: string, enum: [healthy, unhealthy] }
        service: { type: string, example: vision-api }
        version: { type: string, example: "1.0.0" }
        services:
          type: object
          description: Aggregated health from downstream services
          additionalProperties:
            type: object

    ListJobsResponse:
      type: object
      required: [jobs, total, limit, offset]
      properties:
        jobs:
          type: array
          items: { $ref: "#/components/schemas/JobSummary" }
        total:  { type: integer }
        limit:  { type: integer }
        offset: { type: integer }

    JobSummary:
      type: object
      required: [job_id, service, status, progress]
      properties:
        job_id:       { type: string }
        service:      { type: string, enum: [vision, faces, scenes, semantics, objects] }
        status:       { type: string, enum: [queued, processing, completed, failed] }
        progress:     { type: number, minimum: 0, maximum: 1 }
        source:       { type: string, nullable: true }
        source_id:    { type: string, nullable: true }
        created_at:   { type: string, format: date-time, nullable: true }
        started_at:   { type: string, format: date-time, nullable: true }
        completed_at: { type: string, format: date-time, nullable: true }
        result_summary: { type: object, nullable: true }
        results:
          type: object
          nullable: true
          description: Full results; populated only when include_results=true
```

---

## Request Shape

The Vision API uses a **nested `modules` configuration** rather than flat `enable_*` booleans. Each module has its own `enabled` flag and `parameters` map:

```json
{
  "source": "/media/videos/scene_12345.mp4",
  "source_id": "12345",
  "processing_mode": "sequential",
  "modules": {
    "scenes":    { "enabled": true,  "parameters": { "scene_threshold": 27.0 } },
    "faces":     { "enabled": true,  "parameters": { "face_min_confidence": 0.9, "max_faces": 50 } },
    "semantics": { "enabled": false, "parameters": {} },
    "objects":   { "enabled": false, "parameters": {} }
  }
}
```

**Field notes:**

- `source` accepts a local path, an HTTP(S) URL (downloaded to `/tmp/downloads`), or an image file. When the source is an image, `modules.scenes` is auto-disabled.
- `source_id` is the scene identifier used for Stash correlation and caching.
- `modules.scenes` and `modules.faces` default to `enabled: true`; `modules.semantics` and `modules.objects` default to `enabled: false`.
- Parameter keys are forwarded as-is to each service — see the per-service documentation for the accepted keys.
- `processing_mode` currently supports only `sequential`. `parallel` is reserved.

**Auto-enabled scenes:** if `modules.semantics.enabled = true` and `parameters.frame_selection = "scene_based"`, the orchestrator automatically enables `modules.scenes` to supply boundary data.

---

## Functional Details

### Orchestration Flow

```
/vision/analyze (POST)
   │
   ├─ validate source (local files: must exist; URLs: downloaded later)
   ├─ generate job_id (or use provided)
   ├─ enqueue BackgroundTask(process_video_analysis)
   └─ return 202 { job_id, status: queued, services_enabled, processing_mode }

process_video_analysis (async)
   │
   ├─ if source is URL → download_url() → replace request.source with local path
   ├─ if source is image → EXIF normalize, auto-disable scenes
   ├─ write initial metadata to vision:job:{id}:metadata in Redis
   ├─ sequential pipeline:
   │     1. scenes   (if enabled)  → results["scenes"]
   │     2. faces    (if enabled)  → receives scene_boundaries from step 1
   │     3. semantics(if enabled)  → receives scene_boundaries from step 1
   │     4. objects  (if enabled)  → stub response
   ├─ each step calls call_service() which POSTs to the service, polls
   │  {service}/jobs/{id}/status at 2s intervals, then GETs results
   ├─ each poll updates orchestrator progress = base + sub_progress * weight
   └─ on completion: write aggregated results to vision:job:{id}:results
```

**Progress model:** weights are divided evenly across enabled modules (`1 / N` per module). Within each module the orchestrator advances linearly based on the sub-service's reported progress.

### Service Coordination

`call_service()` (in `vision-api/app/main.py`) handles the generic submit-poll-retrieve loop for every downstream service. Scene requests hit `/scenes/detect`; all other modules hit `/{service}/analyze`. Sub-jobs inherit IDs from the orchestrator job: `scenes-{job_id}`, `faces-{job_id}`, etc.

Scene boundaries from `results["scenes"]["scenes"]` are flattened to `[{start_timestamp, end_timestamp}, …]` before being forwarded to faces and semantics — semantics builds them independently of whether faces ran, so the two downstream steps are order-agnostic.

### Cross-Namespace Job Lookups

`GET /vision/jobs/{id}/status` and `GET /vision/jobs/{id}/results` search the following Redis namespaces in order:

```
vision → faces → scenes → semantics → objects
```

The first match wins. This lets clients query any job ID (orchestrator or single-service) through the unified `/vision/jobs/…` endpoint.

### Job Listing and Counting

`GET /vision/jobs` aggregates jobs from all service namespaces via `CacheManager.list_jobs()` and deduplicates by `cache_key` (vision-level jobs are preferred). Filters: `status`, `service`, `source_id`, `source`, `start_date`, `end_date`. Pagination: `limit` (1-500, default 50), `offset`. Pass `include_results=true` to embed full result payloads (heavy — use with small page sizes).

`GET /vision/jobs/count` is an optimized count-only variant used by the jobs-viewer UI for tab badges and pagination headers. Returns `{ total, by_service: { vision, faces, scenes, semantics, objects } }`.

### Health Aggregation

`GET /vision/health` fans out to `{scenes,faces,semantics,objects}/{name}/health` (5s timeout) and returns:

```json
{
  "status": "healthy",
  "service": "vision-api",
  "version": "1.0.0",
  "services": {
    "scenes":    { "status": "healthy", ... },
    "faces":     { "status": "healthy", "model": "buffalo_l", ... },
    "semantics": { "status": "healthy", "classifier_loaded": true, "taxonomy": { "loaded": true, "tag_count": 492 } },
    "objects":   { "status": "healthy", "implemented": false, "phase": 4 }
  }
}
```

A downstream failure does not fail the overall response — individual services are marked `unhealthy` with an `error` field.

---

## Examples

### Scenes + Faces (sequential)

```bash
# Submit
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "modules": {
      "scenes": { "enabled": true, "parameters": { "scene_threshold": 27.0 } },
      "faces":  { "enabled": true, "parameters": { "face_min_confidence": 0.9, "max_faces": 50 } }
    }
  }'
# → { "job_id": "…", "status": "queued", "services_enabled": {...}, "processing_mode": "sequential" }

# Poll
curl http://localhost:5010/vision/jobs/{job_id}/status

# Fetch
curl http://localhost:5010/vision/jobs/{job_id}/results
```

### Semantics with auto-enabled scenes

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source": "/media/videos/scene_12345.mp4",
    "source_id": "12345",
    "modules": {
      "semantics": {
        "enabled": true,
        "parameters": {
          "frame_selection": "scene_based",
          "frames_per_scene": 16,
          "min_confidence": 0.75,
          "top_k_tags": 30
        }
      }
    }
  }'
```

`modules.scenes` is auto-enabled because `frame_selection=scene_based` needs boundary data.

### Semantics with sprite-sheet frames (no scene decoding)

```bash
curl -X POST http://localhost:5010/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "12345",
    "modules": {
      "semantics": {
        "enabled": true,
        "parameters": { "frame_selection": "sprite_sheet" }
      }
    }
  }'
```

Sprite URLs and promotional description are fetched from Stash via `source_id`. No frame-server decoding is needed.

### Filtered job listing

```bash
# Completed semantics jobs from the last 24h, full results embedded
curl "http://localhost:5010/vision/jobs?service=semantics&status=completed&include_results=true&limit=10"

# Cheap count for the jobs-viewer header
curl "http://localhost:5010/vision/jobs/count?status=processing"
```

---

## Aggregated Result Shape

```jsonc
{
  "job_id":    "…",
  "source_id": "12345",
  "status":    "completed",

  "scenes": {
    "status": "completed",
    "scenes": [
      { "scene_number": 0, "start_timestamp": 0.0, "end_timestamp": 30.03, "duration": 30.03 }
    ],
    "metadata": { "processing_time_seconds": 3.39 }
  },

  "faces": {
    "status": "completed",
    "faces":  [ { "face_id": "face_0", "embedding": [/* 512-D */], "detections": [ /* … */ ] } ],
    "metadata": {
      "unique_faces": 11,
      "total_detections": 16,
      "processing_time_seconds": 7.05
    }
  },

  "semantics": {
    "status": "completed",
    "semantics": {
      "tags": [
        { "tag_id": "123", "tag_name": "Indoor",       "score": 0.95, "path": "Setting/Indoor",              "decode_type": "hierarchical" },
        { "tag_id": "456", "tag_name": "Living Room",  "score": 0.88, "path": "Setting/Indoor/Living Room",  "decode_type": "hierarchical" }
      ],
      "frame_captions": [
        { "frame_index": 0, "timestamp": 5.0, "caption": "A woman sits on a beige couch…" }
      ],
      "scene_summary":  "A woman relaxes in a warmly decorated living room…",
      "scene_embedding": [0.012, -0.034, 0.078 /* … */]
    },
    "metadata": {
      "classifier_model": "text-only",
      "frames_captioned": 16,
      "taxonomy_size": 492,
      "processing_time_seconds": 45.2
    }
  },

  "objects": { "status": "not_implemented" },

  "metadata": {
    "processing_time_seconds": 56.61,
    "processing_mode": "sequential",
    "services_used": { "scenes": true, "faces": true, "semantics": true, "objects": false }
  }
}
```

---

## Configuration

Environment variables (all optional; defaults shown):

```bash
# Downstream service URLs
SCENES_SERVICE_URL=http://scenes-service:5002
FACES_SERVICE_URL=http://faces-service:5003
SEMANTICS_SERVICE_URL=http://semantics-service:5004
OBJECTS_SERVICE_URL=http://objects-service:5005

# Redis
REDIS_URL=redis://redis:6379/0
CACHE_TTL=31536000          # 1 year (job metadata retention)

# Confidence defaults (overridable per request)
FACES_MIN_CONFIDENCE=0.9
FACES_MIN_QUALITY=0.0
FACES_ENHANCEMENT_QUALITY_TRIGGER=0.5
SCENES_THRESHOLD=27.0
SEMANTICS_MIN_CONFIDENCE=0.75
OBJECTS_MIN_CONFIDENCE=0.5
CLASSIFIER_MODEL=text-only

# Logging
LOG_LEVEL=INFO
```

---

## Related Documentation

- [How to Use](HOW_TO_USE.md)
- [Scenes Service](SCENES_SERVICE.md)
- [Faces Service](FACES_SERVICE.md)
- [Semantics Service](SEMANTICS_SERVICE.md)
- [Objects Service](OBJECTS_SERVICE.md)
- [Resource Manager](RESOURCE_MANAGER.md)
- [Testing Guide](TESTING.md)

---

**Last Updated:** 2026-04-08
**Version:** 3.0.0
**Status:** Phase 3 Complete - Scenes + Faces + Semantics orchestration
