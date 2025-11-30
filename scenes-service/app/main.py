"""
Scenes Service - Main Application
FastAPI server for scene boundary detection
"""

import os
import uuid
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from .models import (
    DetectScenesRequest,
    DetectJobResponse,
    DetectJobStatus,
    DetectJobResults,
    SceneBoundary,
    VideoMetadata,
    HealthResponse,
    JobStatus,
    ErrorResponse
)
from .cache_manager import CacheManager
from .scene_detector import SceneDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
OPENCV_DEVICE = os.getenv("OPENCV_DEVICE", "cuda")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Global instances
cache_manager: Optional[CacheManager] = None
scene_detector: Optional[SceneDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initialize services on startup, cleanup on shutdown
    """
    global cache_manager, scene_detector

    logger.info("Starting Scenes Service...")

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="scenes", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize scene detector
    scene_detector = SceneDetector(opencv_device=OPENCV_DEVICE)
    detector_info = scene_detector.get_detector_info()
    logger.info(f"Scene detector initialized: {detector_info}")

    yield

    # Cleanup
    logger.info("Shutting down Scenes Service...")
    if cache_manager:
        await cache_manager.disconnect()
    logger.info("Scenes Service stopped")


app = FastAPI(
    title="Scenes Service",
    description="Scene boundary detection service using PySceneDetect",
    version="1.0.0",
    lifespan=lifespan
)


async def process_detection_job(
    job_id: str,
    cache_key: str,
    request: DetectScenesRequest
):
    """
    Background task to process scene detection

    Args:
        job_id: Job identifier
        cache_key: Content-based cache key
        request: Detection request parameters
    """
    try:
        logger.info(f"Starting job {job_id}")

        # Update status to processing
        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.PROCESSING.value,
            progress=0.0,
            stage="initializing"
        )

        start_time = time.time()

        # Get video info
        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.PROCESSING.value,
            progress=0.1,
            stage="analyzing_video"
        )

        duration, fps, total_frames = scene_detector.get_video_info(request.video_path)

        # Detect scenes
        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.PROCESSING.value,
            progress=0.2,
            stage="detecting_scenes",
            message="Analyzing video content..."
        )

        scenes_raw = scene_detector.detect_scenes(
            video_path=request.video_path,
            detection_method=request.detection_method.value,
            scene_threshold=request.scene_threshold,
            min_scene_length=request.min_scene_length
        )

        processing_time = time.time() - start_time

        # Convert to SceneBoundary objects
        scenes = [
            SceneBoundary(
                scene_number=scene_num,
                start_frame=start_frame,
                end_frame=end_frame,
                start_timestamp=start_time,
                end_timestamp=end_time,
                duration=end_time - start_time
            )
            for scene_num, (start_frame, end_frame, start_time, end_time) in enumerate(scenes_raw)
        ]

        # If no scenes detected, create a default scene spanning entire video
        if not scenes:
            logger.info("No scene boundaries detected, creating default scene for entire video")
            scenes = [
                SceneBoundary(
                    scene_number=0,
                    start_frame=0,
                    end_frame=total_frames - 1,
                    start_timestamp=0.0,
                    end_timestamp=duration,
                    duration=duration
                )
            ]

        # Create metadata
        metadata = VideoMetadata(
            video_path=request.video_path,
            detection_method=request.detection_method.value,
            total_frames=total_frames,
            video_duration_seconds=duration,
            video_fps=fps,
            processing_time_seconds=processing_time
        )

        # Create results
        results = DetectJobResults(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            cache_key=cache_key,
            scenes=scenes,
            metadata=metadata
        )

        # Cache results
        await cache_manager.cache_job_results(
            job_id=job_id,
            cache_key=cache_key,
            results=results.dict(),
            ttl=request.cache_duration
        )

        # Update final status
        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=1.0,
            stage="completed",
            message=f"Detected {len(scenes)} scenes in {processing_time:.2f}s"
        )

        logger.info(f"Job {job_id} completed: {len(scenes)} scenes in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.FAILED.value,
            progress=0.0,
            error=str(e)
        )


@app.post("/detect", response_model=DetectJobResponse, status_code=202)
async def detect_scenes(
    request: DetectScenesRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit scene detection job

    Args:
        request: Detection request parameters
        background_tasks: FastAPI background task manager

    Returns:
        Job submission response with job_id and status
    """
    try:
        # Validate video path
        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {request.video_path}"
            )

        # Generate cache key
        params = request.dict(exclude={"job_id", "cache_duration"})
        cache_key = cache_manager.generate_cache_key(request.video_path, params)

        # Check cache
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)

        if cached_job_id:
            logger.info(f"Cache hit: returning existing job {cached_job_id}")

            # Return existing job
            metadata = await cache_manager.get_job_metadata(cached_job_id)

            return DetectJobResponse(
                job_id=cached_job_id,
                status=JobStatus(metadata.get("status", "completed")),
                created_at=metadata.get("created_at", ""),
                cache_key=cache_key,
                estimated_scenes=metadata.get("estimated_scenes")
            )

        # Create new job
        job_id = request.job_id or str(uuid.uuid4())

        # Estimate scene count
        duration, fps, total_frames = scene_detector.get_video_info(request.video_path)
        estimated_scenes = scene_detector.estimate_scene_count(duration)

        # Create job metadata
        metadata = {
            "job_id": job_id,
            "cache_key": cache_key,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "video_path": request.video_path,
            "estimated_scenes": estimated_scenes
        }

        # Cache metadata
        await cache_manager.cache_job_metadata(
            job_id=job_id,
            cache_key=cache_key,
            metadata=metadata,
            ttl=request.cache_duration
        )

        # Queue background task
        background_tasks.add_task(process_detection_job, job_id, cache_key, request)

        logger.info(f"Job {job_id} queued (cache_key: {cache_key[:16]}...)")

        return DetectJobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=metadata["created_at"],
            cache_key=cache_key,
            estimated_scenes=estimated_scenes
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=DetectJobStatus)
async def get_job_status(job_id: str):
    """
    Get job status and progress

    Args:
        job_id: Job identifier

    Returns:
        Job status information
    """
    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        # Count scenes if completed
        scenes_detected = None

        if metadata.get("status") == JobStatus.COMPLETED.value:
            results = await cache_manager.get_job_results(job_id)
            if results:
                scenes_detected = len(results.get("scenes", []))

        return DetectJobStatus(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            scenes_detected=scenes_detected,
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


@app.get("/jobs/{job_id}/results", response_model=DetectJobResults)
async def get_job_results(job_id: str):
    """
    Get job results (only available when status=completed)

    Args:
        job_id: Job identifier

    Returns:
        Complete job results with scene boundaries and metadata
    """
    try:
        # Check job status
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (status: {metadata['status']})"
            )

        # Get results
        results = await cache_manager.get_job_results(job_id)

        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return DetectJobResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check

    Returns:
        Health status and service information
    """
    try:
        # Check Redis connection
        cache_size_mb = await cache_manager.get_cache_size()
        active_jobs = await cache_manager.count_active_jobs()

        # Check GPU availability
        gpu_available = OPENCV_DEVICE == "cuda"

        # Get detector info
        detector_info = scene_detector.get_detector_info()

        return HealthResponse(
            status="healthy",
            service="scenes-service",
            version="1.0.0",
            detection_methods=detector_info["available_detectors"],
            gpu_available=gpu_available,
            active_jobs=active_jobs,
            cache_size_mb=cache_size_mb
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        Error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": str(exc),
                "type": type(exc).__name__
            }
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
