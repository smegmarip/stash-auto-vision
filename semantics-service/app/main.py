"""
Semantics Service - Main Application
FastAPI server for tag classification using trained multi-view classifier.

Pipeline: frame extraction → JoyCaption beta-one → LLM summary → tag classifier
Replaces the old SigLIP zero-shot service and JoyCaption captioning service.
"""

import gc
import io
import os
import time
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import httpx

from .models import (
    AnalyzeSemanticsRequest,
    AnalyzeSemanticsResponse,
    JobStatusResponse,
    SceneContext,
    SemanticsOutcome,
    SemanticsResults,
    SemanticsMetadata,
    SceneMetadata,
    ClassifierTag,
    FrameCaptionResult,
    HealthResponse,
    TaxonomyStatus,
    JobStatus,
    FrameSelectionMethod,
    SemanticsOperation,
)
from .cache_manager import CacheManager
from .classifier import TagClassifier
from .caption_generator import CaptionGenerator
from .llama_runtime import LlamaRuntime
from .summary_generator import SummaryGenerator
from .title_generator import TitleGenerator
from .taxonomy_builder import TaxonomyBuilder
from .frame_client import FrameServerClient
from .gpu_client import GPUClient, LeaseState
from .scenes_client import ScenesServerClient
from .model_manager import ModelManager
from .job_queue import JobQueue
from .worker import SemanticsWorker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_SERVER_URL = os.getenv("FRAME_SERVER_URL", "http://frame-server:5001")
SCENES_SERVER_URL = os.getenv("SCENES_SERVER_URL", "http://scenes-service:5002")
RESOURCE_MANAGER_URL = os.getenv("RESOURCE_MANAGER_URL", "http://resource-manager:5007")
STASH_URL = os.getenv("STASH_URL", "")
STASH_API_KEY = os.getenv("STASH_API_KEY", "")
SEMANTICS_TAG_ID = os.getenv("SEMANTICS_TAG_ID", "")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "text-only")
CLASSIFIER_DEVICE = os.getenv("CLASSIFIER_DEVICE", "cuda")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------
cache_manager: Optional[CacheManager] = None
tag_classifier: Optional[TagClassifier] = None
frame_client: Optional[FrameServerClient] = None
gpu_client: Optional[GPUClient] = None
_classifier_lease_id: Optional[str] = None
_gpu_lease_id: Optional[str] = None  # Long-lived lease: max(joycaption, bnb_leak + llama)
_restart_pending: bool = False  # Set when eviction/expiry requires container restart
llama_runtime: Optional[LlamaRuntime] = None
summary_generator: Optional[SummaryGenerator] = None
title_generator: Optional[TitleGenerator] = None
caption_generator: Optional[CaptionGenerator] = None
model_manager: Optional[ModelManager] = None
job_queue: Optional[JobQueue] = None
worker: Optional[SemanticsWorker] = None
taxonomy_data: Optional[dict] = None
taxonomy_status: TaxonomyStatus = TaxonomyStatus(loaded=False)

MODEL_IDLE_TIMEOUT = int(os.getenv("SEMANTICS_MODEL_IDLE_TIMEOUT", "300"))
JOB_LOCK_TTL = int(os.getenv("SEMANTICS_JOB_LOCK_TTL", "3600"))


async def _load_taxonomy_background():
    """Background task: fetch taxonomy from Stash and initialize classifier tag cache."""
    global taxonomy_data, taxonomy_status, tag_classifier

    if not STASH_URL:
        logger.warning("STASH_URL not set — taxonomy must be provided via custom_taxonomy parameter")
        return

    try:
        logger.info(f"Loading taxonomy from {STASH_URL} (root_tag_id={SEMANTICS_TAG_ID or 'all'})")
        builder = TaxonomyBuilder()
        taxonomy_data = await builder.build_from_stash(
            stash_url=STASH_URL,
            stash_api_key=STASH_API_KEY,
            root_tag_id=SEMANTICS_TAG_ID or None,
        )
        tag_count = len(taxonomy_data.get("tags", []))
        logger.info(f"Taxonomy loaded: {tag_count} tags")

        if tag_classifier and tag_classifier.is_checkpoint_ready:
            tag_classifier.load_taxonomy(taxonomy_data)
            logger.info("Classifier tag cache rebuilt from taxonomy")

        taxonomy_status.loaded = True
        taxonomy_status.tag_count = tag_count
        taxonomy_status.source = "stash"
        taxonomy_status.last_loaded = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    except Exception as e:
        logger.error(f"Failed to load taxonomy from Stash: {e}", exc_info=True)
        taxonomy_status.loaded = False
        taxonomy_status.source = f"error: {str(e)[:100]}"




# Single GPU lease covering the entire semantics pipeline:
# JoyCaption BnB NF4 leaks ALL its VRAM on unload (~7 GB measured).
# peak = max(joycaption, bnb_leak + llama) = max(7GB, 7GB + 2.5GB) ≈ 10GB
GPU_LEASE_VRAM_MB = 10000
GPU_LEASE_TTL = 6 * 3600  # 6 hours — consecutive jobs reuse BnB leaked VRAM


async def _evict_gpu_lease(lease_id: str) -> bool:
    """Eviction callback for the GPU lease.
    Triggers a graceful container restart since the bitsandbytes VRAM
    leak can only be reclaimed by process exit.
    Sets _restart_pending; actual restart is triggered from the
    /resources/{lease_id}/status endpoint when polled by the
    resource manager."""
    global _restart_pending
    logger.info(f"GPU lease {lease_id} evicted — scheduling container restart")
    _restart_pending = True
    return True


async def _check_restart_needed():
    """Check if a restart is pending and no jobs are active."""
    global _restart_pending
    if not _restart_pending:
        return
    # Only restart if no lease is busy
    if gpu_client:
        for rec in gpu_client.get_all_leases().values():
            if rec.state.value == "busy":
                return
    logger.info("No busy leases — restarting container to reclaim bitsandbytes VRAM")
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, cleanup on shutdown."""
    global cache_manager, tag_classifier, frame_client, gpu_client, _classifier_lease_id, model_manager
    global llama_runtime, summary_generator, title_generator, caption_generator, model_manager
    global job_queue, worker

    logger.info("Starting Semantics Service v2.0 (tag classifier pipeline)")

    # Cache manager
    cache_manager = CacheManager(REDIS_URL, module="semantics", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Job queue (uses cache_manager's Redis connection)
    job_queue = JobQueue(redis=cache_manager.redis, module="semantics", lock_ttl_seconds=JOB_LOCK_TTL)
    logger.info(f"Job queue initialized (worker_id={job_queue.worker_id}, lock_ttl={JOB_LOCK_TTL}s)")

    # Frame server client
    frame_client = FrameServerClient(FRAME_SERVER_URL)
    logger.info("Frame server client initialized")

    # GPU client for lease management
    gpu_client = GPUClient(
        resource_manager_url=RESOURCE_MANAGER_URL,
        service_name="semantics-service",
        service_url=f"http://semantics-service:{os.getenv('SEMANTICS_PORT', '5004')}",
    )
    await gpu_client.announce_startup()
    logger.info("GPU client initialized")

    # Caption generator (JoyCaption, 4-bit NF4 — fits 15.6GB VRAM alongside classifier)
    # Training pipeline used bfloat16 on H100 (80GB), we must quantize on 16GB cards.
    caption_generator = CaptionGenerator(use_quantization=True, device=CLASSIFIER_DEVICE)
    logger.info("Caption generator initialized (4-bit quantization)")

    # Shared Llama runtime (Llama 3.1 8B — exclusive with caption)
    # Will be quantized on CUDA to fit in VRAM after caption is unloaded.
    # Both summary and title generators share this single loaded model.
    llama_runtime = LlamaRuntime(device=CLASSIFIER_DEVICE)
    summary_generator = SummaryGenerator(llm=llama_runtime)
    title_generator = TitleGenerator(llm=llama_runtime)
    logger.info(f"Llama runtime initialized (model={llama_runtime.model_name}, device={llama_runtime.device})")

    # Reserve VRAM for the classifier before loading it (perpetual — never evicted)
    CLASSIFIER_VRAM_MB = 3800
    try:
        _classifier_lease_id = await gpu_client.lease(vram_mb=CLASSIFIER_VRAM_MB, priority=1, perpetual=True)
        if _classifier_lease_id:
            logger.info(f"Perpetual GPU lease acquired for classifier ({CLASSIFIER_VRAM_MB} MB)")
    except Exception as e:
        logger.warning(f"Could not acquire perpetual GPU lease: {e}")

    # Tag classifier (~3.8GB with BGE backbone — kept loaded)
    tag_classifier = TagClassifier(model_variant=CLASSIFIER_MODEL, device=CLASSIFIER_DEVICE)
    try:
        tag_classifier.load_model()
        logger.info(f"Tag classifier model loaded: {CLASSIFIER_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}", exc_info=True)
        logger.warning("Classifier will be unavailable until model is loaded")

    # Model manager with idle-timeout unloading
    model_manager = ModelManager(idle_timeout=MODEL_IDLE_TIMEOUT)
    # Both models are exclusive — even quantized, they can't safely coexist
    # alongside the classifier in 16GB VRAM. The classifier (~1.4GB) stays
    # loaded separately. 4-bit JoyCaption ~7GB, 8-bit Llama 8B ~10GB.
    # The Llama runtime is registered once and shared by both the summary
    # and title generators (same weights, different prompts).
    model_manager.register("caption", caption_generator, vram_mb=7000, exclusive=True)
    model_manager.register("llm", llama_runtime, vram_mb=10000, exclusive=True)
    model_manager.start_cleanup_loop()
    logger.info(f"Model manager initialized (idle_timeout={MODEL_IDLE_TIMEOUT}s)")

    # Load taxonomy in background (non-blocking startup)
    asyncio.create_task(_load_taxonomy_background())

    # Start the worker — single consumer of the job queue
    worker = SemanticsWorker(queue=job_queue, process_fn=_run_pipeline)
    await worker.start()

    yield

    # Cleanup
    logger.info("Shutting down Semantics Service...")
    if worker:
        await worker.stop()
    if model_manager:
        model_manager.stop_cleanup_loop()
        model_manager.unload_all()
    if cache_manager:
        await cache_manager.disconnect()
    if tag_classifier:
        tag_classifier.unload()
    if gpu_client:
        await gpu_client.close()
    logger.info("Semantics Service stopped")


app = FastAPI(
    title="Semantics Service",
    description="Tag classification using trained multi-view classifier",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

async def _load_custom_taxonomy(custom_taxonomy) -> dict:
    """Load taxonomy from custom_taxonomy parameter (URL or inline tags array)."""
    builder = TaxonomyBuilder()

    if isinstance(custom_taxonomy, str):
        # URL — fetch and parse
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(custom_taxonomy)
            resp.raise_for_status()
            data = resp.json()
        # Accept either raw tags array or findTags wrapper
        if isinstance(data, dict) and "data" in data:
            tags = data["data"]["findTags"]["tags"]
        elif isinstance(data, dict) and "tags" in data:
            tags = data["tags"]
        elif isinstance(data, list):
            tags = data
        else:
            raise ValueError("Unrecognized taxonomy JSON format")
    elif isinstance(custom_taxonomy, list):
        tags = custom_taxonomy
    else:
        raise ValueError(f"custom_taxonomy must be a URL string or list of tag dicts, got {type(custom_taxonomy)}")

    return builder.build_from_tags(tags, root_tag_id=SEMANTICS_TAG_ID or None)


async def _fetch_scene_from_stash(scene_id: str) -> Dict[str, Any]:
    """Fetch scene data from Stash via findScene GraphQL query."""
    if not STASH_URL:
        raise RuntimeError("STASH_URL not set — cannot fetch scene data")

    query = """
    query FindScene($id: ID!) {
        findScene(id: $id) {
            id
            title
            details
            paths { sprite vtt }
            performers { id name gender }
            files { path duration width height frame_rate }
        }
    }
    """
    graphql_url = f"{STASH_URL.rstrip('/')}/graphql"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if STASH_API_KEY:
        headers["ApiKey"] = STASH_API_KEY

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(graphql_url, json={"query": query, "variables": {"id": scene_id}}, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    if "errors" in body and body["errors"]:
        raise RuntimeError(f"Stash GraphQL errors: {body['errors']}")

    scene = body.get("data", {}).get("findScene")
    if not scene:
        raise RuntimeError(f"Scene {scene_id} not found in Stash")

    return scene


def _build_scene_context(scene_id: str, stash_scene: Optional[Dict[str, Any]], request: AnalyzeSemanticsRequest) -> SceneContext:
    """Build SceneContext from Stash data, with request parameters as overrides."""
    params = request.parameters

    # Start from Stash data (or empty)
    if stash_scene:
        paths = stash_scene.get("paths") or {}
        primary_file = (stash_scene.get("files") or [{}])[0]
        performers = stash_scene.get("performers") or []
        ctx = SceneContext(
            scene_id=scene_id,
            source=primary_file.get("path") or request.source,
            title=stash_scene.get("title"),
            sprite_image_url=paths.get("sprite"),
            sprite_vtt_url=paths.get("vtt"),
            details=stash_scene.get("details"),
            duration=primary_file.get("duration") or 0,
            frame_rate=primary_file.get("frame_rate"),
            width=primary_file.get("width"),
            height=primary_file.get("height"),
            performer_count=len(performers),
            performer_genders=[p.get("gender") for p in performers if p.get("gender")],
        )
    else:
        ctx = SceneContext(scene_id=scene_id, source=request.source)

    # Request parameters override Stash data
    if params.sprite_image_url is not None:
        ctx.sprite_image_url = params.sprite_image_url
    if params.sprite_vtt_url is not None:
        ctx.sprite_vtt_url = params.sprite_vtt_url
    if params.details is not None:
        ctx.details = params.details
    if request.source:
        ctx.source = request.source

    return ctx


async def _extract_frame_images(
    ctx: SceneContext,
    params,
    scene_boundaries: Optional[List[Dict]],
) -> Tuple[List[Image.Image], List[float], List[str]]:
    """Extract frames from sprite sheets (default) or video via frame-server.

    Returns:
        Tuple of (images, timestamps, file_paths) where timestamps are in seconds
        and file_paths are local paths for subprocess use.
    """
    use_sprites = params.frame_selection == FrameSelectionMethod.SPRITE_SHEET and ctx.sprite_vtt_url and ctx.sprite_image_url

    if use_sprites:
        frames_data = await frame_client.extract_sprites(
            video_path=ctx.source,
            sprite_vtt_url=ctx.sprite_vtt_url,
            sprite_image_url=ctx.sprite_image_url,
            frames_per_scene=params.frames_per_scene,
            scene_boundaries=scene_boundaries,
        )
    else:
        if params.frame_selection == FrameSelectionMethod.SPRITE_SHEET:
            logger.warning("Sprite sheet requested but sprite URLs not available, falling back to video frames")
        frames_data = await frame_client.extract_frames(
            video_path=ctx.source,
            sampling_interval=params.sampling_interval,
            scene_boundaries=scene_boundaries,
            frames_per_scene=params.frames_per_scene,
            select_sharpest=params.select_sharpest,
            candidate_multiplier=params.sharpness_candidate_multiplier,
            min_quality=params.min_frame_quality,
        )

    if not frames_data:
        raise RuntimeError("No frames extracted")

    # Fetch actual frame images, timestamps, and file paths
    frames = frames_data if isinstance(frames_data, list) else frames_data.get("frames", [])
    images = []
    timestamps = []
    file_paths = []
    for frame_info in frames:
        try:
            url = frame_info.get("url") if isinstance(frame_info, dict) else None
            if not url:
                continue
            if url.startswith("file://"):
                path = url[7:]
                img = Image.open(path).convert("RGB")
                file_paths.append(path)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                # Save to temp file so subprocess can access it
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp/frames", delete=False)
                img.save(tmp, format="JPEG", quality=95)
                file_paths.append(tmp.name)
                tmp.close()
            images.append(img)
            timestamps.append(frame_info.get("timestamp", 0.0) if isinstance(frame_info, dict) else 0.0)
        except Exception as e:
            logger.warning(f"Failed to load frame {frame_info.get('index', '?') if isinstance(frame_info, dict) else '?'}: {e}")

    if not images:
        raise RuntimeError("Failed to load any frame images")

    return images, timestamps, file_paths


def _resolve_operations(params) -> set:
    """Return the set of active operation names from SemanticsParameters."""
    ops = params.operations
    if not ops or SemanticsOperation.ALL in ops:
        return {"title", "summary", "tags"}
    return {op.value for op in ops}


async def _run_pipeline(job_id: str, request_payload: dict):
    """Worker callback: full tag classification pipeline for a single job.

    Called by SemanticsWorker after acquiring the job from the Redis queue.
    Job sequencing is enforced by the queue's atomic Lua script — this
    function trusts that only one instance runs at a time.
    """
    global taxonomy_data, taxonomy_status

    # Deserialize the stored request
    request = AnalyzeSemanticsRequest.model_validate(request_payload)

    start_time = time.time()
    lease_id = None

    try:
        logger.info(f"Starting semantics analysis job {job_id} for scene {request.source_id}")
        params = request.parameters
        active_ops = _resolve_operations(params)

        # Initialize job metadata
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        cache_params = {
            "model": params.model_variant,
            "min_confidence": params.min_confidence,
            "top_k": params.top_k_tags,
            "frames_per_scene": params.frames_per_scene,
            "hierarchical": params.use_hierarchical_decoding,
            "operations": sorted(active_ops),
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)

        metadata = {
            "job_id": job_id,
            "status": JobStatus.PROCESSING.value,
            "progress": 0.0,
            "stage": "initializing",
            "message": "Initializing pipeline",
            "created_at": now,
            "started_at": now,
            "source_id": request.source_id,
            "source": request.source,
            "cache_key": cache_key,
        }
        await cache_manager.cache_job_metadata(job_id, cache_key, metadata)

        # --- Step 0: Taxonomy ---
        active_taxonomy = taxonomy_data
        if request.custom_taxonomy:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.02, stage="loading_taxonomy", message="Loading custom taxonomy")
            active_taxonomy = await _load_custom_taxonomy(request.custom_taxonomy)
            logger.info(f"Custom taxonomy loaded: {len(active_taxonomy.get('tags', []))} tags")

        if not active_taxonomy or not active_taxonomy.get("tags"):
            raise RuntimeError("No taxonomy available. Set STASH_URL or provide custom_taxonomy.")

        # Ensure classifier has this taxonomy loaded
        if tag_classifier and tag_classifier.is_loaded:
            existing = tag_classifier.taxonomy
            if existing is not active_taxonomy:
                existing_ids = {str(t["id"]) for t in (existing or {}).get("tags", [])}
                new_ids = {str(t["id"]) for t in active_taxonomy.get("tags", [])}
                if existing_ids != new_ids:
                    logger.info(f"Taxonomy changed ({len(existing_ids)} → {len(new_ids)} tags), reloading classifier")
                    tag_classifier.load_taxonomy(active_taxonomy)
        elif tag_classifier and tag_classifier.is_checkpoint_ready:
            tag_classifier.load_taxonomy(active_taxonomy)
        elif not tag_classifier or not tag_classifier.is_checkpoint_ready:
            raise RuntimeError("Tag classifier checkpoint not ready")

        # --- Step 1: Resolve scene context from Stash ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.03, stage="fetching_scene", message="Fetching scene data from Stash")
        stash_scene = None
        try:
            stash_scene = await _fetch_scene_from_stash(request.source_id)
            logger.info(f"Fetched scene {request.source_id} from Stash")
        except Exception as e:
            logger.warning(f"Could not fetch scene from Stash: {e}")

        ctx = _build_scene_context(request.source_id, stash_scene, request)
        logger.info(f"Scene context: source={ctx.source}, sprites={'yes' if ctx.sprite_vtt_url else 'no'}, promo={'yes' if ctx.has_promo else 'no'}, duration={ctx.duration:.0f}s")

        # --- Step 1b: Fetch scene boundaries ---
        scene_boundaries = params.scene_boundaries
        if not scene_boundaries and request.scenes_job_id:
            try:
                scenes_client = ScenesServerClient(SCENES_SERVER_URL)
                scene_boundaries = await scenes_client.get_scene_boundaries(request.scenes_job_id)
                logger.info(f"Retrieved {len(scene_boundaries)} scene boundaries")
            except Exception as e:
                logger.warning(f"Failed to fetch scene boundaries: {e}")

        # --- Step 2: Extract frames ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.05, stage="extracting_frames", message="Extracting frames")
        frame_images, frame_timestamps, frame_paths = await _extract_frame_images(ctx, params, scene_boundaries)
        num_frames = len(frame_images)
        logger.info(f"Extracted {num_frames} frames")

        # Pad or truncate to exactly 16 frames (classifier requirement)
        target_frames = 16
        if num_frames < target_frames:
            logger.warning(f"Only {num_frames} frames extracted, padding to {target_frames}")
            while len(frame_images) < target_frames:
                frame_images.append(frame_images[-1])
                frame_timestamps.append(frame_timestamps[-1] if frame_timestamps else 0.0)
                frame_paths.append(frame_paths[-1] if frame_paths else "")
        elif num_frames > target_frames:
            import numpy as np
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
            frame_images = [frame_images[i] for i in indices]
            frame_timestamps = [frame_timestamps[i] for i in indices]
            frame_paths = [frame_paths[i] for i in indices]

        # --- Step 3: Request GPU lease ---
        # Single long-lived lease (6h) covering peak(JoyCaption, BnB_leak + Llama).
        # Persists after the job to cover the BnB VRAM leak. Consecutive
        # jobs reuse it. On eviction or expiry, container restarts to
        # reclaim the leaked VRAM.
        global _gpu_lease_id
        await cache_manager.update_job_status(job_id, JobStatus.WAITING_FOR_GPU.value, progress=0.10, stage="requesting_gpu", message="Requesting GPU access")
        if gpu_client and not _gpu_lease_id:
            try:
                _gpu_lease_id = await gpu_client.lease(
                    vram_mb=GPU_LEASE_VRAM_MB, priority=3, timeout_seconds=600,
                    ttl=GPU_LEASE_TTL,
                )
                if _gpu_lease_id:
                    gpu_client.on_evict(_gpu_lease_id, _evict_gpu_lease)
                    logger.info(f"GPU lease acquired: {_gpu_lease_id} ({GPU_LEASE_VRAM_MB} MB, {GPU_LEASE_TTL//3600}h TTL)")
            except Exception as e:
                logger.warning(f"GPU lease request failed: {e}. Proceeding without lease.")
        if _gpu_lease_id and gpu_client:
            gpu_client.mark_busy(_gpu_lease_id)

        # --- Step 4: Generate captions ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.15, stage="captioning", message="Generating frame captions with JoyCaption")

        def _generate_captions():
            with model_manager.using("caption"):
                return caption_generator.generate_captions(frame_images)

        raw_captions = await asyncio.to_thread(_generate_captions)
        fixed_captions = [CaptionGenerator.fix_caption(c, i) for i, c in enumerate(raw_captions)]
        logger.info(f"Generated {len(fixed_captions)} captions")

        # --- Step 5: Generate scene summary ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.60, stage="summarizing", message="Generating scene summary")

        frame_caption_dicts = [{"frame_index": i, "timestamp": frame_timestamps[i], "caption": c} for i, c in enumerate(fixed_captions)]

        def _generate_summary():
            with model_manager.using("llm"):
                return summary_generator.generate_summary(
                    frame_caption_dicts, ctx.promo_desc, ctx.duration,
                    ctx.performer_count, ctx.performer_genders, ctx.resolution,
                )

        try:
            scene_summary = await asyncio.to_thread(_generate_summary)
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}. Using concatenated captions as fallback.", exc_info=True)
            scene_summary = " ".join(c for c in fixed_captions)

        logger.info(f"Summary generated ({len(scene_summary)} chars)")

        # --- Step 6: Generate suggested title (reuses loaded Llama) ---
        suggested_title: Optional[str] = None
        if "title" in active_ops:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.80, stage="titling", message="Generating scene title")

            def _generate_title():
                with model_manager.using("llm"):
                    return title_generator.generate_title(
                        scene_source=request.source, scene_summary=scene_summary,
                        promo_desc=ctx.promo_desc, duration=ctx.duration,
                        performer_count=ctx.performer_count,
                        performer_genders=ctx.performer_genders,
                        resolution=ctx.resolution,
                    )

            try:
                suggested_title = await asyncio.to_thread(_generate_title)
            except Exception as e:
                logger.warning(f"Title generation failed: {e}. Continuing without suggested_title.", exc_info=True)

            if suggested_title:
                logger.info(f"Suggested title: {suggested_title!r}")
        else:
            logger.info("Skipping title generation (not in operations)")

        # --- Step 7: Run tag classifier ---
        tags = []
        scene_embedding = None
        if "tags" in active_ops:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.85, stage="classifying", message="Running tag classifier")

            prediction = await asyncio.to_thread(
                tag_classifier.predict,
                frame_captions=fixed_captions,
                summary=scene_summary,
                promo_desc=ctx.promo_desc,
                has_promo=ctx.has_promo,
                top_k=params.top_k_tags,
                min_score=params.min_confidence,
                use_hierarchical_decoding=params.use_hierarchical_decoding,
            )

            tags = prediction["tags"]
            logger.info(f"Classifier returned {len(tags)} tags")

            # Optional: scene embedding
            if params.generate_embeddings:
                try:
                    scene_embedding = tag_classifier.get_scene_embedding(fixed_captions, scene_summary, ctx.promo_desc, ctx.has_promo)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
        else:
            logger.info("Skipping tag classification (not in operations)")

        # --- Build results ---
        processing_time = time.time() - start_time

        classifier_tags = [
            ClassifierTag(
                tag_id=t["tag_id"],
                tag_name=t["tag_name"],
                score=round(t["score"], 4),
                path=t.get("path", ""),
                decode_type=t.get("decode_type", "direct"),
            )
            for t in tags
        ]

        frame_caption_results = [
            FrameCaptionResult(frame_index=i, timestamp=frame_timestamps[i], caption=c)
            for i, c in enumerate(fixed_captions)
        ]

        outcome = SemanticsOutcome(
            tags=classifier_tags if "tags" in active_ops else [],
            frame_captions=frame_caption_results,
            scene_summary=scene_summary if "summary" in active_ops else None,
            suggested_title=suggested_title if "title" in active_ops else None,
            scene_embedding=scene_embedding if "tags" in active_ops else None,
        )

        tag_name_to_id = {str(t.get("name", "")): str(t["id"]) for t in active_taxonomy.get("tags", []) if t.get("name")}

        result_metadata = SemanticsMetadata(
            source=ctx.source,
            source_id=request.source_id,
            total_frames_extracted=num_frames,
            frames_captioned=len(fixed_captions),
            classifier_model=params.model_variant,
            processing_time_seconds=round(processing_time, 2),
            device=CLASSIFIER_DEVICE,
            taxonomy_size=len(active_taxonomy.get("tags", [])),
            has_promo=ctx.has_promo,
            sprite_image_url=ctx.sprite_image_url,
            sprite_vtt_url=ctx.sprite_vtt_url,
            tag_name_to_id=tag_name_to_id,
            scene=SceneMetadata(
                title=ctx.title,
                duration=ctx.duration if ctx.duration else None,
                resolution=ctx.resolution,
                frame_rate=ctx.frame_rate,
                performer_count=ctx.performer_count,
                performer_genders=ctx.performer_genders,
            ),
        )

        results = {
            "job_id": job_id,
            "source_id": request.source_id,
            "status": JobStatus.COMPLETED.value,
            "semantics": outcome.model_dump(),
            "metadata": result_metadata.model_dump(),
        }

        await cache_manager.cache_job_results(job_id, cache_key, results)
        await cache_manager.update_job_status(job_id, JobStatus.COMPLETED.value, progress=1.0, stage="completed", message=f"Analysis complete in {processing_time:.1f}s ({len(tags)} tags)")

        logger.info(f"Job {job_id} completed in {processing_time:.1f}s — {len(tags)} tags predicted")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        try:
            await cache_manager.update_job_status(job_id, JobStatus.FAILED.value, progress=0.0, stage="failed", message=f"Job failed: {str(e)[:200]}", error=str(e))
        except Exception:
            pass

    finally:
        # GPU lease persists (BnB leak). Mark idle so it's evictable.
        if _gpu_lease_id and gpu_client:
            gpu_client.mark_idle(_gpu_lease_id)
        gc.collect()

        # Check if eviction/expiry requires container restart
        await _check_restart_needed()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/semantics/analyze", response_model=AnalyzeSemanticsResponse, status_code=202)
async def analyze_semantics(request: AnalyzeSemanticsRequest):
    """Submit tag classification analysis job to the Redis-backed queue.

    The endpoint never touches GPU code. It stores the request payload in
    Redis and pushes the job_id to the pending queue. The worker (a single
    background task) pulls jobs serially using an atomic Lua script.
    """
    try:
        # Validate source exists (for local paths)
        if request.source and not request.source.startswith(("http://", "https://")) and not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        if not job_queue:
            raise HTTPException(status_code=503, detail="Job queue not initialized")

        # Check cache
        cache_params = {
            "model": request.parameters.model_variant,
            "min_confidence": request.parameters.min_confidence,
            "top_k": request.parameters.top_k_tags,
            "frames_per_scene": request.parameters.frames_per_scene,
            "hierarchical": request.parameters.use_hierarchical_decoding,
            "operations": sorted(_resolve_operations(request.parameters)),
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)

        if cached_job_id:
            logger.info(f"Cache hit for {request.source}: {cached_job_id}")
            if request.job_id and request.job_id != cached_job_id:
                await cache_manager.create_job_alias(request.job_id, cached_job_id)
            return AnalyzeSemanticsResponse(
                job_id=request.job_id or cached_job_id,
                status=JobStatus.COMPLETED,
                message="Results retrieved from cache",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                cache_hit=True,
            )

        job_id = request.job_id or str(uuid.uuid4())

        # Store the full request payload in Redis (source of truth for the worker)
        await job_queue.store_request(job_id, request.model_dump())

        # Push to pending queue — the worker will pick it up via atomic acquire
        pending_count = await job_queue.enqueue(job_id)

        logger.info(f"Job {job_id} enqueued for scene {request.source_id} (pending={pending_count})")
        return AnalyzeSemanticsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message=f"Tag classification job queued ({pending_count} ahead in queue)",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status."""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            created_at=metadata["created_at"],
            started_at=metadata.get("started_at"),
            completed_at=metadata.get("completed_at"),
            error=metadata.get("error"),
            gpu_wait_position=metadata.get("gpu_wait_position"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/results", response_model=SemanticsResults)
async def get_job_results(job_id: str):
    """Get job results."""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(status_code=409, detail=f"Job not completed (status: {metadata['status']})")

        results = await cache_manager.get_job_results(job_id)
        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/{lease_id}/status")
async def get_lease_status(lease_id: str):
    """Return internal lease state for resource-manager revocation protocol.
    Also triggers pending restart if the lease is not busy."""
    if not gpu_client:
        raise HTTPException(status_code=503, detail="GPU client not initialized")
    rec = gpu_client.get_lease(lease_id)
    if not rec:
        return {"lease_id": lease_id, "state": "unknown"}

    # If a restart is pending and this lease isn't busy, restart now
    if _restart_pending and rec.state != LeaseState.BUSY:
        asyncio.create_task(_check_restart_needed())

    return rec.to_dict()


@app.post("/resources/{lease_id}/release")
async def release_lease(lease_id: str):
    """Attempt to release a lease (called by resource-manager for eviction)."""
    if not gpu_client:
        raise HTTPException(status_code=503, detail="GPU client not initialized")
    rec = gpu_client.get_lease(lease_id)
    if not rec:
        return {"accepted": False, "reason": "Lease not found"}
    if gpu_client.is_busy(lease_id):
        return {"accepted": False, "reason": "Lease is busy"}
    if rec.perpetual:
        return {"accepted": False, "reason": "Cannot evict perpetual lease"}
    asyncio.create_task(gpu_client.evict(lease_id))
    return {"accepted": True}


@app.get("/semantics/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    classifier_loaded = tag_classifier is not None and tag_classifier.is_loaded
    queue_stats = None
    if job_queue:
        try:
            queue_stats = await job_queue.queue_stats()
        except Exception as e:
            logger.warning(f"Failed to fetch queue stats: {e}")
    return HealthResponse(
        status="healthy" if classifier_loaded and taxonomy_status.loaded else "degraded",
        classifier_model=CLASSIFIER_MODEL if classifier_loaded else None,
        classifier_loaded=classifier_loaded,
        device=CLASSIFIER_DEVICE,
        taxonomy=taxonomy_status,
        default_min_confidence=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.75")),
        queue=queue_stats,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
