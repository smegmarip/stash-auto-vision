"""
Semantics Service - Main Application
FastAPI server for semantic scene classification using CLIP/SigLIP
"""

import os
import uuid
import time
import asyncio
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from collections import Counter

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from .models import (
    AnalyzeSemanticsRequest,
    AnalyzeSemanticsResponse,
    JobStatusResponse,
    SceneSemanticsOutcome,
    SemanticsResults,
    FrameSemantics,
    SemanticTag,
    SceneSemanticSummary,
    SemanticsMetadata,
    HealthResponse,
    JobStatus,
)
from .cache_manager import CacheManager
from .clip_classifier import CLIPClassifier
from .frame_client import FrameServerClient
from .scenes_client import ScenesServerClient

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CLIP_MODEL = os.getenv("CLIP_MODEL", "google/siglip-base-patch16-224")
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cuda")
FRAME_SERVER_URL = os.getenv("FRAME_SERVER_URL", "http://frame-server:5001")
SCENES_SERVER_URL = os.getenv("SCENES_SERVER_URL", "http://scenes-service:5002")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
STUB_MODE = os.getenv("SEMANTICS_STUB_MODE", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
cache_manager: Optional[CacheManager] = None
clip_classifier: Optional[CLIPClassifier] = None
frame_client: Optional[FrameServerClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initialize services on startup, cleanup on shutdown
    """
    global cache_manager, clip_classifier, frame_client

    logger.info("Starting Semantics Service...")

    if STUB_MODE:
        logger.warning("Running in STUB MODE - not loading CLIP model")
        yield
        return

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="semantics", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize CLIP classifier
    clip_classifier = CLIPClassifier(model_name=CLIP_MODEL, device=CLIP_DEVICE)
    clip_classifier.load_model()
    model_info = clip_classifier.get_model_info()
    logger.info(f"CLIP classifier initialized: {model_info}")

    # Initialize frame-server client
    frame_client = FrameServerClient(FRAME_SERVER_URL)
    logger.info("Frame server client initialized")

    yield

    # Cleanup
    logger.info("Shutting down Semantics Service...")
    if cache_manager:
        await cache_manager.disconnect()
    if clip_classifier:
        clip_classifier.cleanup()
    logger.info("Semantics Service stopped")


app = FastAPI(
    title="Semantics Service",
    description="Semantic scene classification using CLIP/SigLIP",
    version="1.0.0",
    lifespan=lifespan
)


async def process_semantics_analysis(job_id: str, request: AnalyzeSemanticsRequest):
    """
    Background task to process semantics analysis

    Args:
        job_id: Job identifier
        request: Analysis request
    """
    start_time = time.time()

    try:
        logger.info(f"Starting semantics analysis job {job_id}")

        # Initialize job metadata
        metadata = {
            "job_id": job_id,
            "status": JobStatus.PROCESSING.value,
            "progress": 0.0,
            "stage": "extracting_frames",
            "message": "Extracting frames from video",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source_id": request.source_id,
            "source": request.source,
        }

        # Generate cache key
        cache_params = {
            "model": request.parameters.model,
            "tags": request.parameters.classification_tags or [],
            "prompts": request.parameters.custom_prompts or [],
            "min_confidence": request.parameters.min_confidence,
            "top_k": request.parameters.top_k_tags,
            "sampling_interval": request.parameters.sampling_interval,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)
        metadata["cache_key"] = cache_key

        await cache_manager.cache_job_metadata(job_id, cache_key, metadata)

        # Step 0: Fetch scene boundaries (if scenes_job_id provided)
        scene_boundaries = request.parameters.scene_boundaries

        if not scene_boundaries and request.scenes_job_id:
            logger.info(f"Fetching scene boundaries from scenes job: {request.scenes_job_id}")
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.05,
                stage="fetching_scenes",
                message="Fetching scene boundaries from scenes-service"
            )

            try:
                scenes_client = ScenesServerClient(SCENES_SERVER_URL)
                scenes_result = await scenes_client.get_scene_boundaries(request.scenes_job_id)
                scene_boundaries = scenes_result
                logger.info(f"Retrieved {len(scene_boundaries)} scene boundaries")
            except Exception as e:
                logger.warning(f"Failed to fetch scene boundaries: {e}. Continuing without scene boundaries.")
                scene_boundaries = None

        # Step 1: Extract frames
        logger.info(f"Extracting frames from: {request.source}")
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.1,
            stage="extracting_frames",
            message="Extracting frames from video"
        )

        frame_extraction_result = await frame_client.extract_frames(
            video_path=request.source,
            sampling_interval=request.parameters.sampling_interval
        )
        frame_metadata = frame_extraction_result.frames if frame_extraction_result else None

        if not frame_metadata:
            raise RuntimeError("No frames extracted from video")

        logger.info(f"Extracted {len(frame_metadata)} frames")

        # Step 2: Load frame images
        await cache_manager.update_job_status(
            job_id,
            JobStatus.PROCESSING.value,
            progress=0.3,
            stage="loading_frames",
            message=f"Loading {len(frame_metadata)} frame images"
        )

        timestamps = [frame.timestamp for frame in frame_metadata]
        frames = await frame_client.get_frames_batch(request.source, timestamps)

        # Filter out None frames (failed to load)
        valid_frames = []
        valid_metadata = []
        for frame, meta in zip(frames, frame_metadata):
            if frame is not None:
                valid_frames.append(frame)
                valid_metadata.append(meta)

        if not valid_frames:
            raise RuntimeError("Failed to load any frames")

        logger.info(f"Loaded {len(valid_frames)}/{len(frame_metadata)} frames successfully")

        # Step 3: Classify frames (if tags provided)
        frame_results = []

        if request.parameters.classification_tags or request.parameters.custom_prompts:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.4,
                stage="classifying",
                message="Classifying frames with CLIP/SigLIP"
            )

            # Combine predefined tags and custom prompts
            all_tags = (request.parameters.classification_tags or []) + (request.parameters.custom_prompts or [])

            classification_results = clip_classifier.classify_frames(
                frames=valid_frames,
                tags=all_tags,
                batch_size=request.parameters.batch_size,
                min_confidence=request.parameters.min_confidence,
                top_k=request.parameters.top_k_tags
            )

            # Build frame results with tags
            for meta, class_result in zip(valid_metadata, classification_results):
                frame_result = {
                    "frame_index": meta.index,
                    "timestamp": meta.timestamp,
                    "tags": class_result["tags"],
                    "embedding": None  # Will add if embeddings requested
                }
                frame_results.append(frame_result)

        else:
            # No tags provided, just metadata
            for meta in valid_metadata:
                frame_result = {
                    "frame_index": meta.index,
                    "timestamp": meta.timestamp,
                    "tags": [],
                    "embedding": None
                }
                frame_results.append(frame_result)

        # Step 4: Generate embeddings (if requested)
        if request.parameters.generate_embeddings:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.7,
                stage="generating_embeddings",
                message="Generating scene embeddings"
            )

            embeddings = clip_classifier.generate_embeddings(
                frames=valid_frames,
                batch_size=request.parameters.batch_size,
                normalize=True
            )

            # Add embeddings to frame results
            for frame_result, embedding in zip(frame_results, embeddings):
                frame_result["embedding"] = embedding.tolist()

        # Step 5: Aggregate scene summaries (if scene boundaries provided)
        scene_summaries = None
        if scene_boundaries:
            await cache_manager.update_job_status(
                job_id,
                JobStatus.PROCESSING.value,
                progress=0.9,
                stage="aggregating",
                message="Aggregating scene summaries"
            )

            scene_summaries = aggregate_scene_semantics(
                frame_results,
                scene_boundaries,
                request.parameters.min_confidence
            )

        # Build final results
        processing_time = time.time() - start_time
        total_frames = 0
        if frame_extraction_result:
            total_frames = frame_extraction_result.metadata.total_frames

        result_metadata = SemanticsMetadata(
            source=request.source,
            source_type="video",
            total_frames=total_frames,
            model=request.parameters.model,
            frames_analyzed=len(frame_results),
            processing_time_seconds=processing_time,
            device=CLIP_DEVICE,
            batch_size=request.parameters.batch_size,
            total_tags_generated=sum(len(f["tags"]) for f in frame_results)
        )

        outcome = SceneSemanticsOutcome(
            frames=frame_results,
            scene_summaries=scene_summaries,
            metadata=result_metadata
        )

        results = {
            "job_id": job_id,
            "source_id": request.source_id,
            "status": JobStatus.COMPLETED.value,
            "semantics": outcome.dict(),
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
            message=f"Analysis complete in {processing_time:.1f}s"
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
        # Explicitly clean up large arrays to prevent memory issues
        if 'valid_frames' in locals():
            del valid_frames
        if 'frames' in locals():
            del frames
        if 'embeddings' in locals():
            del embeddings

        # Force garbage collection for CPU mode
        import gc
        gc.collect()


def aggregate_scene_semantics(
    frame_results: List[Dict[str, Any]],
    scene_boundaries: List[Dict[str, float]],
    min_confidence: float
) -> List[Dict[str, Any]]:
    """
    Aggregate per-frame tags into scene-level summaries

    Args:
        frame_results: List of frame results with tags
        scene_boundaries: Scene boundaries (start_timestamp, end_timestamp)
        min_confidence: Minimum confidence threshold

    Returns:
        List of scene summaries
    """
    scene_summaries = []

    for scene in scene_boundaries:
        start = scene["start_timestamp"]
        end = scene["end_timestamp"]

        # Find frames within this scene
        scene_frames = [
            f for f in frame_results
            if start <= f["timestamp"] <= end
        ]

        if not scene_frames:
            continue

        # Count tag frequencies
        tag_counts = Counter()
        confidence_sum = {}
        confidence_count = {}

        for frame in scene_frames:
            for tag_obj in frame["tags"]:
                tag = tag_obj["tag"]
                confidence = tag_obj["confidence"]

                if confidence >= min_confidence:
                    tag_counts[tag] += 1

                    if tag not in confidence_sum:
                        confidence_sum[tag] = 0.0
                        confidence_count[tag] = 0

                    confidence_sum[tag] += confidence
                    confidence_count[tag] += 1

        # Get most common tags
        dominant_tags = [tag for tag, count in tag_counts.most_common(5)]

        # Calculate average confidence across all tags in scene
        avg_confidence = 0.0
        if confidence_count:
            total_conf = sum(confidence_sum.values())
            total_count = sum(confidence_count.values())
            avg_confidence = total_conf / total_count if total_count > 0 else 0.0

        scene_summaries.append({
            "start_timestamp": start,
            "end_timestamp": end,
            "dominant_tags": dominant_tags,
            "frame_count": len(scene_frames),
            "avg_confidence": avg_confidence
        })

    return scene_summaries


@app.post("/semantics/analyze", response_model=AnalyzeSemanticsResponse, status_code=202)
async def analyze_semantics(request: AnalyzeSemanticsRequest, background_tasks: BackgroundTasks):
    """Submit semantics analysis job"""

    if STUB_MODE:
        logger.info("Received semantics analysis request (stub mode)")
        return AnalyzeSemanticsResponse(
            job_id=f"semantics-stub-{int(time.time())}",
            status=JobStatus.NOT_IMPLEMENTED,
            message="Semantics analysis module is not yet implemented (Phase 2)",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    try:
        # Validate video exists
        if not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        # Generate cache key
        cache_params = {
            "model": request.parameters.model,
            "tags": request.parameters.classification_tags or [],
            "prompts": request.parameters.custom_prompts or [],
            "min_confidence": request.parameters.min_confidence,
            "top_k": request.parameters.top_k_tags,
            "sampling_interval": request.parameters.sampling_interval,
        }
        cache_key = cache_manager.generate_cache_key(request.source, cache_params)

        # Check cache
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)
        if cached_job_id:
            logger.info(f"Cache hit for {request.source}: {cached_job_id}")
            return AnalyzeSemanticsResponse(
                job_id=cached_job_id,
                status=JobStatus.COMPLETED,
                message="Results retrieved from cache",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                cache_hit=True
            )

        # Generate new job ID
        job_id = request.job_id or str(uuid.uuid4())

        # Queue background task
        background_tasks.add_task(process_semantics_analysis, job_id, request)

        logger.info(f"Semantics analysis job {job_id} queued")

        return AnalyzeSemanticsResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            message="Semantics analysis job queued",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            cache_hit=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""

    if STUB_MODE:
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status": JobStatus.NOT_IMPLEMENTED.value,
                "progress": 0.0,
                "message": "Semantics analysis module is not yet implemented (Phase 2)",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
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
            error=metadata.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/jobs/{job_id}/results", response_model=SemanticsResults)
async def get_job_results(job_id: str):
    """Get job results"""

    if STUB_MODE:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Semantics analysis module is not yet implemented (Phase 2)",
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

        return SemanticsResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantics/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    if STUB_MODE:
        return HealthResponse(
            status="healthy",
            service="semantics-service",
            version="1.0.0",
            implemented=False,
            phase=2,
            message="Stub service - awaiting CLIP integration",
            model=None,
            device=None,
            default_min_confidence=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.5"))
        )

    return HealthResponse(
        status="healthy",
        service="semantics-service",
        version="1.0.0",
        implemented=True,
        phase=2,
        message="CLIP/SigLIP semantic classification active",
        model=CLIP_MODEL,
        device=CLIP_DEVICE,
        default_min_confidence=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.5"))
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
