"""
Captioning Service - Main Application
FastAPI server for video captioning using JoyCaption VLM
"""

import os
import uuid
import time
import asyncio
import gc
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from .models import (
    AnalyzeCaptionsRequest,
    AnalyzeCaptionsResponse,
    JobStatusResponse,
    CaptionResults,
    CaptionOutcome,
    FrameCaption,
    CaptionTag,
    SceneCaptionSummary,
    CaptionMetadata,
    HealthResponse,
    JobStatus,
    PromptType,
    TaxonomyResponse
)
from .cache_manager import CacheManager
from .joycaption import JoyCaptionProcessor
from .prompt_templates import (
    get_prompt_template,
    parse_booru_tags,
    detect_tag_category,
    estimate_vram_usage
)
from .tag_aligner import TagAligner
from .stash_client import StashClient
from .frame_client import FrameServerClient
from .scenes_client import ScenesServerClient
from .resource_client import ResourceManagerClient

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
FRAME_SERVER_URL = os.getenv("FRAME_SERVER_URL", "http://frame-server:5001")
SCENES_SERVER_URL = os.getenv("SCENES_SERVER_URL", "http://scenes-service:5002")
RESOURCE_MANAGER_URL = os.getenv("RESOURCE_MANAGER_URL", "http://resource-manager:5007")
STASH_URL = os.getenv("STASH_URL", "http://localhost:9999")
STASH_API_KEY = os.getenv("STASH_API_KEY", "")
CAPTION_DEVICE = os.getenv("CAPTION_DEVICE", "cuda")
USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
STUB_MODE = os.getenv("CAPTIONING_STUB_MODE", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
cache_manager: Optional[CacheManager] = None
joycaption: Optional[JoyCaptionProcessor] = None
tag_aligner: Optional[TagAligner] = None
stash_client: Optional[StashClient] = None
frame_client: Optional[FrameServerClient] = None
resource_client: Optional[ResourceManagerClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cache_manager, joycaption, tag_aligner, stash_client, frame_client, resource_client

    logger.info("Starting Captioning Service...")

    if STUB_MODE:
        logger.warning("Running in STUB MODE - model will not be loaded")
        yield
        return

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="captions", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize JoyCaption processor (don't load model yet)
    joycaption = JoyCaptionProcessor(
        device=CAPTION_DEVICE,
        use_quantization=USE_QUANTIZATION
    )
    logger.info("JoyCaption processor initialized (model not loaded)")

    # Initialize clients
    frame_client = FrameServerClient(FRAME_SERVER_URL)
    resource_client = ResourceManagerClient(
        RESOURCE_MANAGER_URL,
        service_name="captioning-service"
    )
    stash_client = StashClient(STASH_URL, STASH_API_KEY)
    logger.info("Service clients initialized")

    # Try to load taxonomy from Stash
    try:
        if await stash_client.health_check():
            taxonomy = await stash_client.get_all_tags()
            tag_aligner = TagAligner(taxonomy)
            logger.info(f"Tag aligner initialized with {len(taxonomy)} tags")
        else:
            logger.warning("Stash not available, tag alignment disabled")
            tag_aligner = TagAligner([])
    except Exception as e:
        logger.warning(f"Failed to load taxonomy from Stash: {e}")
        tag_aligner = TagAligner([])

    yield

    # Cleanup
    logger.info("Shutting down Captioning Service...")

    if joycaption and joycaption.model_loaded:
        joycaption.unload_model()

    if cache_manager:
        await cache_manager.disconnect()

    if frame_client:
        await frame_client.close()

    if resource_client:
        if resource_client.has_gpu_lease:
            await resource_client.release_gpu()
        await resource_client.close()

    if stash_client:
        await stash_client.close()

    logger.info("Captioning Service stopped")


app = FastAPI(
    title="Captioning Service",
    description="Video captioning using JoyCaption VLM",
    version="1.0.0",
    lifespan=lifespan
)


async def process_captioning_job(job_id: str, request: AnalyzeCaptionsRequest):
    """Background task to process captioning analysis"""
    start_time = time.time()
    gpu_wait_start = None
    gpu_wait_time = 0.0

    try:
        logger.info(f"Starting captioning job {job_id}")

        # Initialize job metadata
        metadata = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "stage": "initializing",
            "message": "Job queued",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source_id": request.source_id,
            "source": request.source,
        }

        # Generate cache key
        cache_params = {
            "prompt_type": request.parameters.prompt_type.value,
            "frame_selection": request.parameters.frame_selection.value,
            "min_confidence": request.parameters.min_confidence,
            "max_tags": request.parameters.max_tags_per_frame,
            "align_taxonomy": request.parameters.align_to_taxonomy,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)
        metadata["cache_key"] = cache_key

        await cache_manager.cache_job_metadata(job_id, cache_key, metadata)

        # Step 1: Request GPU access from resource manager
        logger.info("Requesting GPU access...")
        await cache_manager.update_job_status(
            job_id,
            JobStatus.WAITING_FOR_GPU.value,
            progress=0.05,
            stage="waiting_for_gpu",
            message="Waiting for GPU access"
        )

        vram_needed = estimate_vram_usage(
            batch_size=request.parameters.batch_size,
            use_quantization=request.parameters.use_quantization
        ) * 1024  # Convert GB to MB

        gpu_wait_start = time.time()

        try:
            if await resource_client.health_check():
                gpu_request = await resource_client.request_gpu(
                    vram_mb=vram_needed,
                    priority=5
                )

                if not gpu_request.get("granted"):
                    # Wait for GPU
                    await resource_client.wait_for_gpu(
                        gpu_request["request_id"],
                        max_wait=300.0
                    )
            else:
                logger.warning("Resource manager not available, proceeding without GPU management")
        except Exception as e:
            logger.warning(f"GPU management failed: {e}, proceeding anyway")

        gpu_wait_time = time.time() - gpu_wait_start

        # Step 2: Load model if needed
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.1,
            stage="loading_model",
            message="Loading JoyCaption model"
        )

        if not joycaption.model_loaded:
            joycaption.load_model()

        # Step 3: Get scene boundaries if needed
        scene_boundaries = None

        if request.scenes_job_id:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.15,
                stage="fetching_scenes",
                message="Fetching scene boundaries"
            )

            try:
                scenes_client = ScenesServerClient(SCENES_SERVER_URL)
                scene_boundaries = await scenes_client.get_scene_boundaries(request.scenes_job_id)
                logger.info(f"Retrieved {len(scene_boundaries)} scene boundaries")
            except Exception as e:
                logger.warning(f"Failed to fetch scene boundaries: {e}")

        # Step 4: Extract frames
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.2,
            stage="extracting_frames",
            message="Extracting frames from video"
        )

        frame_result = await frame_client.extract_frames(
            video_path=request.source,
            sampling_interval=request.parameters.sampling_interval,
            scene_boundaries=scene_boundaries,
            frames_per_scene=request.parameters.frames_per_scene
        )

        if not frame_result or not frame_result.get("frames"):
            raise RuntimeError("No frames extracted from video")

        frame_metadata = frame_result["frames"]
        logger.info(f"Extracted {len(frame_metadata)} frames")

        # Step 5: Load frame images
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.3,
            stage="loading_frames",
            message=f"Loading {len(frame_metadata)} frames"
        )

        timestamps = [f["timestamp"] for f in frame_metadata]
        frames = await frame_client.get_frames_batch(request.source, timestamps)

        valid_frames = []
        valid_metadata = []
        for frame, meta in zip(frames, frame_metadata):
            if frame is not None:
                valid_frames.append(frame)
                valid_metadata.append(meta)

        if not valid_frames:
            raise RuntimeError("Failed to load any frames")

        logger.info(f"Loaded {len(valid_frames)}/{len(frame_metadata)} frames")

        # Step 6: Generate captions
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.4,
            stage="captioning",
            message=f"Generating captions for {len(valid_frames)} frames"
        )

        prompt = request.parameters.custom_prompt or get_prompt_template(
            request.parameters.prompt_type
        )

        def progress_callback(current, total):
            progress = 0.4 + (0.4 * current / total)
            asyncio.create_task(
                cache_manager.update_job_status(
                    job_id,
                    JobStatus.PROCESSING.value,
                    progress=progress,
                    stage="captioning",
                    message=f"Captioning frame {current}/{total}"
                )
            )

        captions = joycaption.caption_images_batch(
            images=valid_frames,
            prompt=prompt,
            batch_size=request.parameters.batch_size,
            progress_callback=progress_callback
        )

        # Step 7: Parse and align tags
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.85,
            stage="aligning_tags",
            message="Aligning tags to taxonomy"
        )

        frame_results = []
        for meta, caption in zip(valid_metadata, captions):
            # Parse booru-style tags from caption
            raw_tags = parse_booru_tags(caption)

            # Align to taxonomy if enabled
            if request.parameters.align_to_taxonomy and tag_aligner:
                aligned_tags = tag_aligner.align_tags(
                    raw_tags,
                    min_confidence=request.parameters.min_confidence
                )
            else:
                aligned_tags = [
                    CaptionTag(
                        tag=tag,
                        confidence=max(0.5, 1.0 - (i * 0.02)),
                        source="joycaption",
                        category=detect_tag_category(tag)
                    )
                    for i, tag in enumerate(raw_tags[:request.parameters.max_tags_per_frame])
                ]

            # Determine scene index
            scene_index = None
            if scene_boundaries:
                ts = meta["timestamp"]
                for i, scene in enumerate(scene_boundaries):
                    if scene["start_timestamp"] <= ts <= scene["end_timestamp"]:
                        scene_index = i
                        break

            frame_results.append(FrameCaption(
                frame_index=meta.get("index", 0),
                timestamp=meta["timestamp"],
                raw_caption=caption,
                tags=aligned_tags,
                scene_index=scene_index,
                prompt_type_used=request.parameters.prompt_type
            ))

        # Step 8: Aggregate scene summaries
        scene_summaries = None
        if scene_boundaries:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.95,
                stage="aggregating",
                message="Aggregating scene summaries"
            )
            scene_summaries = aggregate_scene_captions(frame_results, scene_boundaries)

        # Build final results
        processing_time = time.time() - start_time
        vram_peak = joycaption.get_vram_usage()[0] if joycaption.model_loaded else None

        result_metadata = CaptionMetadata(
            source=request.source,
            total_frames=frame_result.get("metadata", {}).get("total_frames", 0),
            frames_captioned=len(frame_results),
            model="joycaption",
            model_variant="alpha-two",
            quantization="4-bit" if request.parameters.use_quantization else "none",
            prompt_type=request.parameters.prompt_type,
            processing_time_seconds=processing_time,
            device=CAPTION_DEVICE,
            vram_peak_mb=vram_peak,
            gpu_wait_time_seconds=gpu_wait_time
        )

        outcome = CaptionOutcome(
            frames=frame_results,
            scene_summaries=scene_summaries,
            metadata=result_metadata
        )

        results = {
            "job_id": job_id,
            "source_id": request.source_id,
            "status": JobStatus.COMPLETED.value,
            "captions": outcome.dict(),
            "metadata": result_metadata.dict()
        }

        # Cache results
        await cache_manager.cache_job_results(job_id, cache_key, results)

        # Update final status
        await cache_manager.update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=1.0,
            stage="completed",
            message=f"Captioned {len(frame_results)} frames in {processing_time:.1f}s"
        )

        logger.info(f"Job {job_id} completed in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await cache_manager.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            progress=0.0,
            stage="failed",
            message=f"Job failed: {str(e)[:100]}",
            error=str(e)
        )

    finally:
        # Release GPU if we have a lease
        if resource_client and resource_client.has_gpu_lease:
            try:
                await resource_client.release_gpu()
            except Exception as e:
                logger.warning(f"Failed to release GPU: {e}")

        # Cleanup
        if 'valid_frames' in locals():
            del valid_frames
        if 'frames' in locals():
            del frames
        gc.collect()


def aggregate_scene_captions(
    frame_results: List[FrameCaption],
    scene_boundaries: List[Dict]
) -> List[SceneCaptionSummary]:
    """Aggregate frame captions into scene summaries"""
    from collections import Counter

    summaries = []

    for i, scene in enumerate(scene_boundaries):
        start = scene["start_timestamp"]
        end = scene["end_timestamp"]

        # Find frames in this scene
        scene_frames = [
            f for f in frame_results
            if start <= f.timestamp <= end
        ]

        if not scene_frames:
            continue

        # Count tag frequencies
        tag_counts = Counter()
        total_conf = 0.0
        total_tags = 0

        for frame in scene_frames:
            for tag in frame.tags:
                tag_counts[tag.tag] += 1
                total_conf += tag.confidence
                total_tags += 1

        dominant_tags = [tag for tag, _ in tag_counts.most_common(10)]
        avg_confidence = total_conf / total_tags if total_tags > 0 else 0.0

        # Combine raw captions
        combined = " | ".join(f.raw_caption[:100] for f in scene_frames[:3])

        summaries.append(SceneCaptionSummary(
            scene_index=i,
            start_timestamp=start,
            end_timestamp=end,
            dominant_tags=dominant_tags,
            frame_count=len(scene_frames),
            avg_confidence=avg_confidence,
            combined_caption=combined[:500] if combined else None
        ))

    return summaries


@app.post("/captions/analyze", response_model=AnalyzeCaptionsResponse, status_code=202)
async def analyze_captions(request: AnalyzeCaptionsRequest, background_tasks: BackgroundTasks):
    """Submit captioning analysis job"""

    if STUB_MODE:
        return AnalyzeCaptionsResponse(
            job_id=f"captions-stub-{int(time.time())}",
            status=JobStatus.NOT_IMPLEMENTED,
            message="Captioning service is running in stub mode",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    try:
        # Validate video exists
        if not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        # Generate cache key
        cache_params = {
            "prompt_type": request.parameters.prompt_type.value,
            "frame_selection": request.parameters.frame_selection.value,
            "min_confidence": request.parameters.min_confidence,
            "max_tags": request.parameters.max_tags_per_frame,
            "align_taxonomy": request.parameters.align_to_taxonomy,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)

        # Check cache
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)
        if cached_job_id:
            logger.info(f"Cache hit for {request.source}: {cached_job_id}")

            if request.job_id and request.job_id != cached_job_id:
                await cache_manager.create_job_alias(request.job_id, cached_job_id)

            return AnalyzeCaptionsResponse(
                job_id=request.job_id or cached_job_id,
                status=JobStatus.COMPLETED,
                message="Results retrieved from cache",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                cache_hit=True
            )

        # Generate new job ID
        job_id = request.job_id or f"captions-{uuid.uuid4()}"

        # Queue background task
        background_tasks.add_task(process_captioning_job, job_id, request)

        logger.info(f"Captioning job {job_id} queued")

        return AnalyzeCaptionsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message="Captioning job queued",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""

    if STUB_MODE:
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": JobStatus.NOT_IMPLEMENTED.value,
                "progress": 0.0,
                "message": "Captioning service is running in stub mode"
            }
        )

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
            gpu_wait_position=metadata.get("gpu_wait_position")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/jobs/{job_id}/results", response_model=CaptionResults)
async def get_job_results(job_id: str):
    """Get job results"""

    if STUB_MODE:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Captioning service is running in stub mode",
                    "type": "NotImplementedError"
                }
            }
        )

    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (status: {metadata['status']})"
            )

        results = await cache_manager.get_job_results(job_id)

        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return CaptionResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy():
    """Get current tag taxonomy"""

    if not tag_aligner:
        return TaxonomyResponse(
            tags=[],
            categories=[],
            total_tags=0,
            last_synced=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        )

    stats = tag_aligner.get_taxonomy_stats()

    return TaxonomyResponse(
        tags=tag_aligner.taxonomy,
        categories=list(tag_aligner.category_lookup.keys()),
        total_tags=stats["total_tags"],
        last_synced=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    )


@app.post("/captions/taxonomy/sync")
async def sync_taxonomy():
    """Sync taxonomy from Stash"""
    global tag_aligner

    if STUB_MODE:
        return {"status": "stub_mode", "message": "Running in stub mode"}

    try:
        if not await stash_client.health_check():
            raise HTTPException(status_code=503, detail="Stash not available")

        taxonomy = await stash_client.get_all_tags()
        tag_aligner = TagAligner(taxonomy)

        return {
            "status": "synced",
            "tags_loaded": len(taxonomy),
            "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing taxonomy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/captions/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    if STUB_MODE:
        return HealthResponse(
            status="healthy",
            implemented=False,
            message="Stub service - JoyCaption not loaded",
            model=None,
            model_loaded=False,
            device=None,
            gpu_acquired=False
        )

    model_info = joycaption.get_model_info() if joycaption else {}

    return HealthResponse(
        status="healthy",
        implemented=True,
        message="JoyCaption VLM service active",
        model=model_info.get("name"),
        model_loaded=model_info.get("loaded", False),
        device=model_info.get("device"),
        vram_available_mb=model_info.get("vram_mb"),
        gpu_acquired=resource_client.has_gpu_lease if resource_client else False,
        default_min_confidence=float(os.getenv("CAPTIONS_MIN_CONFIDENCE", "0.5"))
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5006)
