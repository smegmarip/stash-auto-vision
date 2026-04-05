"""
Semantics Service - Main Application
FastAPI server for tag classification using trained multi-view classifier.

Pipeline: frame extraction → JoyCaption beta-one → LLM summary → tag classifier
Replaces the old SigLIP zero-shot service and JoyCaption captioning service.
"""

import gc
import os
import time
import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import httpx

from .models import (
    AnalyzeSemanticsRequest,
    AnalyzeSemanticsResponse,
    JobStatusResponse,
    SemanticsOutcome,
    SemanticsResults,
    SemanticsMetadata,
    ClassifierTag,
    FrameCaptionResult,
    HealthResponse,
    TaxonomyStatus,
    JobStatus,
)
from .cache_manager import CacheManager
from .classifier import TagClassifier
from .caption_generator import CaptionGenerator
from .summary_generator import SummaryGenerator
from .taxonomy_builder import TaxonomyBuilder
from .frame_client import FrameServerClient
from .resource_client import ResourceManagerClient
from .scenes_client import ScenesServerClient

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
resource_client: Optional[ResourceManagerClient] = None
summary_generator: Optional[SummaryGenerator] = None
taxonomy_data: Optional[dict] = None
taxonomy_status: TaxonomyStatus = TaxonomyStatus(loaded=False)


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

        if tag_classifier and tag_classifier.is_loaded:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, cleanup on shutdown."""
    global cache_manager, tag_classifier, frame_client, resource_client, summary_generator

    logger.info("Starting Semantics Service v2.0 (tag classifier pipeline)")

    # Cache manager
    cache_manager = CacheManager(REDIS_URL, module="semantics", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Frame server client
    frame_client = FrameServerClient(FRAME_SERVER_URL)
    logger.info("Frame server client initialized")

    # Resource manager client
    resource_client = ResourceManagerClient(RESOURCE_MANAGER_URL)
    logger.info("Resource manager client initialized")

    # Summary generator (calls external LLM API)
    summary_generator = SummaryGenerator()
    logger.info(f"Summary generator initialized (api_base={summary_generator.api_base})")

    # Tag classifier (load model weights)
    tag_classifier = TagClassifier(model_variant=CLASSIFIER_MODEL, device=CLASSIFIER_DEVICE)
    try:
        tag_classifier.load_model()
        logger.info(f"Tag classifier model loaded: {CLASSIFIER_MODEL}")
    except Exception as e:
        logger.error(f"Failed to load classifier model: {e}", exc_info=True)
        logger.warning("Classifier will be unavailable until model is loaded")

    # Load taxonomy in background (non-blocking startup)
    asyncio.create_task(_load_taxonomy_background())

    yield

    # Cleanup
    logger.info("Shutting down Semantics Service...")
    if cache_manager:
        await cache_manager.disconnect()
    if tag_classifier:
        tag_classifier.unload()
    logger.info("Semantics Service stopped")


app = FastAPI(
    title="Semantics Service",
    description="Tag classification using trained multi-view classifier",
    version="2.0.0",
    lifespan=lifespan,
)


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


async def _extract_frame_images(
    job_id: str,
    request: AnalyzeSemanticsRequest,
    scene_boundaries: Optional[List[Dict]],
) -> List[Image.Image]:
    """Extract frames using frame-server with sharpness selection."""
    params = request.parameters

    if params.frame_selection.value == "sprite_sheet" and params.sprite_vtt_url:
        # Sprite sheet extraction
        frames_data = await frame_client.extract_sprites(
            sprite_image_url=params.sprite_image_url,
            sprite_vtt_url=params.sprite_vtt_url,
            max_frames=params.frames_per_scene,
        )
    else:
        # Video frame extraction
        frames_data = await frame_client.extract_frames(
            video_path=request.source,
            sampling_interval=params.sampling_interval,
            scene_boundaries=scene_boundaries,
            frames_per_scene=params.frames_per_scene,
            select_sharpest=params.select_sharpest,
            sharpness_candidate_multiplier=params.sharpness_candidate_multiplier,
            min_quality=params.min_frame_quality,
        )

    if not frames_data:
        raise RuntimeError("No frames extracted from video")

    # Fetch actual frame images
    images = []
    for frame_info in frames_data:
        try:
            img = await frame_client.get_frame(frame_info["url"])
            if img is not None:
                images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load frame {frame_info.get('index', '?')}: {e}")

    if not images:
        raise RuntimeError("Failed to load any frame images")

    return images


async def process_semantics_analysis(job_id: str, request: AnalyzeSemanticsRequest):
    """Background task: full tag classification pipeline."""
    global taxonomy_data, taxonomy_status

    start_time = time.time()
    lease_id = None
    caption_gen = None

    try:
        logger.info(f"Starting semantics analysis job {job_id} for scene {request.source_id}")
        params = request.parameters

        # Initialize job metadata
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        cache_params = {
            "model": params.model_variant,
            "min_confidence": params.min_confidence,
            "top_k": params.top_k_tags,
            "frames_per_scene": params.frames_per_scene,
            "hierarchical": params.use_hierarchical_decoding,
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
            tag_classifier.load_taxonomy(active_taxonomy)
        else:
            raise RuntimeError("Tag classifier not loaded")

        # --- Step 1: Fetch scene boundaries ---
        scene_boundaries = params.scene_boundaries
        if not scene_boundaries and request.scenes_job_id:
            await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.03, stage="fetching_scenes", message="Fetching scene boundaries")
            try:
                scenes_client = ScenesServerClient(SCENES_SERVER_URL)
                scene_boundaries = await scenes_client.get_scene_boundaries(request.scenes_job_id)
                logger.info(f"Retrieved {len(scene_boundaries)} scene boundaries")
            except Exception as e:
                logger.warning(f"Failed to fetch scene boundaries: {e}")

        # --- Step 2: Extract frames ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.05, stage="extracting_frames", message="Extracting frames from video")
        frame_images = await _extract_frame_images(job_id, request, scene_boundaries)
        num_frames = len(frame_images)
        logger.info(f"Extracted {num_frames} frames")

        # Pad or truncate to exactly 16 frames (classifier requirement)
        target_frames = 16
        if num_frames < target_frames:
            logger.warning(f"Only {num_frames} frames extracted, padding to {target_frames}")
            while len(frame_images) < target_frames:
                frame_images.append(frame_images[-1])
        elif num_frames > target_frames:
            # Linearly sample 16 from available
            import numpy as np
            indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
            frame_images = [frame_images[i] for i in indices]

        # --- Step 3: Request GPU ---
        await cache_manager.update_job_status(job_id, JobStatus.WAITING_FOR_GPU.value, progress=0.10, stage="requesting_gpu", message="Requesting GPU access")
        try:
            gpu_result = await resource_client.request_gpu(service_name="semantics-service", vram_required_mb=8000, priority=3, job_id=job_id)
            if gpu_result.get("granted"):
                lease_id = gpu_result.get("lease_id")
            else:
                request_id = gpu_result.get("request_id")
                if request_id:
                    wait_result = await resource_client.wait_for_gpu(request_id, timeout=300)
                    lease_id = wait_result.get("lease_id")
        except Exception as e:
            logger.warning(f"GPU resource manager unavailable: {e}. Proceeding without lease.")

        # --- Step 4: Generate captions ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.15, stage="captioning", message="Generating frame captions with JoyCaption")

        caption_gen = CaptionGenerator(use_quantization=params.use_quantization, device=CLASSIFIER_DEVICE)
        caption_gen.load()

        raw_captions = caption_gen.generate_captions(frame_images)

        # Fix captions to match training format
        fixed_captions = [CaptionGenerator.fix_caption(c, i) for i, c in enumerate(raw_captions)]
        logger.info(f"Generated {len(fixed_captions)} captions")

        # Unload JoyCaption to free GPU memory
        caption_gen.unload()
        caption_gen = None

        # Heartbeat GPU lease
        if lease_id:
            try:
                await resource_client.heartbeat(lease_id)
            except Exception:
                pass

        # --- Step 5: Generate scene summary ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.60, stage="summarizing", message="Generating scene summary")

        # Build frame caption dicts for the summary generator
        frame_caption_dicts = [{"frame_index": i, "timestamp": i * 2.0, "caption": c} for i, c in enumerate(fixed_captions)]
        promo_desc = params.details or ""
        has_promo = bool(promo_desc.strip())

        try:
            scene_summary = await summary_generator.generate_summary(
                frame_captions=frame_caption_dicts,
                promo_desc=promo_desc,
            )
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}. Using concatenated captions as fallback.")
            scene_summary = " ".join(c for c in fixed_captions)

        logger.info(f"Summary generated ({len(scene_summary)} chars)")

        # --- Step 6: Run tag classifier ---
        await cache_manager.update_job_status(job_id, JobStatus.PROCESSING.value, progress=0.80, stage="classifying", message="Running tag classifier")

        prediction = tag_classifier.predict(
            frame_captions=fixed_captions,
            summary=scene_summary,
            promo_desc=promo_desc,
            has_promo=has_promo,
            top_k=params.top_k_tags,
            min_score=params.min_confidence,
            use_hierarchical_decoding=params.use_hierarchical_decoding,
        )

        tags = prediction["tags"]
        logger.info(f"Classifier returned {len(tags)} tags")

        # Optional: scene embedding
        scene_embedding = None
        if params.generate_embeddings:
            try:
                scene_embedding = tag_classifier.get_scene_embedding(fixed_captions, scene_summary, promo_desc, has_promo)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

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
            FrameCaptionResult(frame_index=i, timestamp=i * 2.0, caption=c)
            for i, c in enumerate(fixed_captions)
        ]

        outcome = SemanticsOutcome(
            tags=classifier_tags,
            frame_captions=frame_caption_results,
            scene_summary=scene_summary,
            scene_embedding=scene_embedding,
        )

        result_metadata = SemanticsMetadata(
            source=request.source,
            source_id=request.source_id,
            total_frames_extracted=num_frames,
            frames_captioned=len(fixed_captions),
            classifier_model=params.model_variant,
            processing_time_seconds=round(processing_time, 2),
            device=CLASSIFIER_DEVICE,
            taxonomy_size=len(active_taxonomy.get("tags", [])),
            has_promo=has_promo,
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
        # Release GPU lease
        if lease_id:
            try:
                await resource_client.release_gpu(lease_id)
            except Exception:
                pass

        # Cleanup VLM if still loaded
        if caption_gen and caption_gen.is_loaded:
            caption_gen.unload()

        gc.collect()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/semantics/analyze", response_model=AnalyzeSemanticsResponse, status_code=202)
async def analyze_semantics(request: AnalyzeSemanticsRequest, background_tasks: BackgroundTasks):
    """Submit tag classification analysis job."""
    try:
        # Validate source exists (for local paths)
        if not request.source.startswith(("http://", "https://")) and not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        # Check cache
        cache_params = {
            "model": request.parameters.model_variant,
            "min_confidence": request.parameters.min_confidence,
            "top_k": request.parameters.top_k_tags,
            "frames_per_scene": request.parameters.frames_per_scene,
            "hierarchical": request.parameters.use_hierarchical_decoding,
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
        background_tasks.add_task(process_semantics_analysis, job_id, request)

        logger.info(f"Semantics analysis job {job_id} queued for scene {request.source_id}")
        return AnalyzeSemanticsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message="Tag classification job queued",
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


@app.get("/semantics/jobs/{job_id}/results")
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


@app.get("/semantics/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    classifier_loaded = tag_classifier is not None and tag_classifier.is_loaded
    return HealthResponse(
        status="healthy" if classifier_loaded and taxonomy_status.loaded else "degraded",
        classifier_model=CLASSIFIER_MODEL if classifier_loaded else None,
        classifier_loaded=classifier_loaded,
        device=CLASSIFIER_DEVICE,
        taxonomy=taxonomy_status,
        default_min_confidence=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.75")),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
