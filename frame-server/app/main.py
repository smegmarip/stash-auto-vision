"""
Frame Server - Main Application
FastAPI server for video frame extraction and caching
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
    ExtractFramesRequest,
    ExtractJobResponse,
    ExtractJobStatus,
    ExtractJobResults,
    FrameMetadata,
    VideoMetadata,
    HealthResponse,
    JobStatus,
    ErrorResponse
)
from .cache_manager import CacheManager
from .frame_extractor import FrameExtractor
from .sprite_parser import SpriteParser

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
ENABLE_FALLBACK = os.getenv("ENABLE_FALLBACK", "true").lower() == "true"

# Global instances
cache_manager: Optional[CacheManager] = None
frame_extractor: Optional[FrameExtractor] = None
sprite_parser: Optional[SpriteParser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initialize services on startup, cleanup on shutdown
    """
    global cache_manager, frame_extractor, sprite_parser

    logger.info("Starting Frame Server...")

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="frame", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize frame extractor
    extraction_method = "opencv_cuda" if OPENCV_DEVICE == "cuda" else "opencv_cpu"
    frame_extractor = FrameExtractor(
        extraction_method=extraction_method,
        enable_fallback=ENABLE_FALLBACK
    )
    logger.info(f"Frame extractor initialized (method: {extraction_method}, fallback: {ENABLE_FALLBACK})")

    # Initialize sprite parser
    sprite_parser = SpriteParser()
    logger.info("Sprite parser initialized")

    yield

    # Cleanup
    logger.info("Shutting down Frame Server...")
    if cache_manager:
        await cache_manager.disconnect()
    logger.info("Frame Server stopped")


app = FastAPI(
    title="Frame Server",
    description="Video frame extraction and caching service",
    version="1.0.0",
    lifespan=lifespan
)


async def process_extraction_job(
    job_id: str,
    cache_key: str,
    request: ExtractFramesRequest
):
    """
    Background task to process frame extraction

    Args:
        job_id: Job identifier
        cache_key: Content-based cache key
        request: Extraction request parameters
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

        # Determine extraction method
        use_sprites = request.use_sprites and request.sprite_vtt_url and request.sprite_image_url

        frames_data = []

        if use_sprites:
            logger.info(f"Job {job_id}: Using sprite extraction")

            await cache_manager.update_job_status(
                job_id,
                status=JobStatus.PROCESSING.value,
                progress=0.1,
                stage="downloading_sprites"
            )

            # Process sprites
            frames_data = await sprite_parser.process_sprites(
                sprite_vtt_url=request.sprite_vtt_url,
                sprite_image_url=request.sprite_image_url,
                job_id=job_id,
                output_format=request.output_format.value,
                quality=request.quality
            )

            extraction_method = "sprites"

        else:
            logger.info(f"Job {job_id}: Using frame extraction")

            await cache_manager.update_job_status(
                job_id,
                status=JobStatus.PROCESSING.value,
                progress=0.1,
                stage="extracting_frames"
            )

            # Extract frames
            frames_data = frame_extractor.extract(
                video_path=request.video_path,
                job_id=job_id,
                sampling_interval=request.sampling_strategy.interval_seconds or 2.0,
                scene_boundaries=request.scene_boundaries,
                output_format=request.output_format.value,
                quality=request.quality
            )

            extraction_method = request.extraction_method.value

        # Get video metadata
        duration, fps, total_frames = frame_extractor.get_video_info(request.video_path)

        processing_time = time.time() - start_time

        # Convert frame data to FrameMetadata objects
        frames = [
            FrameMetadata(
                index=idx,
                timestamp=timestamp,
                url=f"file://{file_path}",  # Local file URL
                width=width,
                height=height
            )
            for idx, timestamp, file_path, width, height in frames_data
        ]

        # Create metadata
        metadata = VideoMetadata(
            video_path=request.video_path,
            extraction_method=extraction_method,
            total_frames=total_frames,
            video_duration_seconds=duration,
            video_fps=fps,
            processing_time_seconds=processing_time
        )

        # Create results
        results = ExtractJobResults(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            cache_key=cache_key,
            frames=frames,
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
            message=f"Extracted {len(frames)} frames in {processing_time:.2f}s"
        )

        logger.info(f"Job {job_id} completed: {len(frames)} frames in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.FAILED.value,
            progress=0.0,
            error=str(e)
        )


@app.post("/extract", response_model=ExtractJobResponse, status_code=202)
async def extract_frames(
    request: ExtractFramesRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit frame extraction job

    Args:
        request: Extraction request parameters
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

            return ExtractJobResponse(
                job_id=cached_job_id,
                status=JobStatus(metadata.get("status", "completed")),
                created_at=metadata.get("created_at", ""),
                cache_key=cache_key,
                estimated_frames=metadata.get("estimated_frames")
            )

        # Create new job
        job_id = request.job_id or str(uuid.uuid4())

        # Estimate frame count
        if request.scene_boundaries:
            estimated_frames = len(request.scene_boundaries) * 3  # 3 frames per scene
        else:
            duration, fps, total_frames = frame_extractor.get_video_info(request.video_path)
            interval = request.sampling_strategy.interval_seconds or 2.0
            estimated_frames = int(duration / interval)

        # Create job metadata
        metadata = {
            "job_id": job_id,
            "cache_key": cache_key,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "video_path": request.video_path,
            "estimated_frames": estimated_frames
        }

        # Cache metadata
        await cache_manager.cache_job_metadata(
            job_id=job_id,
            cache_key=cache_key,
            metadata=metadata,
            ttl=request.cache_duration
        )

        # Queue background task
        background_tasks.add_task(process_extraction_job, job_id, cache_key, request)

        logger.info(f"Job {job_id} queued (cache_key: {cache_key[:16]}...)")

        return ExtractJobResponse(
            job_id=job_id,
            status=JobStatus.QUEUED,
            created_at=metadata["created_at"],
            cache_key=cache_key,
            estimated_frames=estimated_frames
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=ExtractJobStatus)
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

        # Count frames if completed
        frames_extracted = None
        frames_total = metadata.get("estimated_frames")

        if metadata.get("status") == JobStatus.COMPLETED.value:
            results = await cache_manager.get_job_results(job_id)
            if results:
                frames_extracted = len(results.get("frames", []))

        return ExtractJobStatus(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            frames_extracted=frames_extracted,
            frames_total=frames_total,
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


@app.get("/jobs/{job_id}/results", response_model=ExtractJobResults)
async def get_job_results(job_id: str):
    """
    Get job results (only available when status=completed)

    Args:
        job_id: Job identifier

    Returns:
        Complete job results with frames and metadata
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

        return ExtractJobResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/extract-frame")
async def extract_single_frame(
    video_path: str,
    timestamp: float,
    output_format: str = "jpeg",
    quality: int = 95
):
    """
    Extract a single frame from video at specific timestamp

    Lightweight endpoint for one-off frame extraction without job tracking.
    Useful for thumbnails, previews, and poster frames.

    Uses automatic fallback across extraction methods (OpenCV → PyAV → FFmpeg)
    for robust handling of corrupted or poorly-encoded videos.

    Args:
        video_path: Absolute path to video file
        timestamp: Timestamp in seconds
        output_format: jpeg or png
        quality: JPEG quality (1-100)

    Returns:
        FileResponse with extracted frame
    """
    try:
        from fastapi.responses import FileResponse

        # Validate video exists
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {video_path}"
            )

        # Use fallback extraction (robust)
        job_id = f"single_{uuid.uuid4().hex[:8]}"
        result = frame_extractor.extract_single_frame_with_fallback(
            video_path=video_path,
            timestamp=timestamp,
            job_id=job_id,
            idx=0,
            output_format=output_format,
            quality=quality
        )

        if result is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract frame at timestamp {timestamp}s (all methods failed)"
            )

        idx, ts, frame_path, width, height = result

        # Get FPS for frame number calculation
        try:
            duration, fps, total_frames = frame_extractor.get_video_info(video_path)
            frame_number = int(timestamp * fps)
        except Exception:
            frame_number = 0

        # Return frame and schedule cleanup
        return FileResponse(
            frame_path,
            media_type=f"image/{output_format}",
            headers={
                "X-Timestamp": str(timestamp),
                "X-Frame-Number": str(frame_number),
                "X-Resolution": f"{width}x{height}"
            },
            background=None  # File will be cleaned by cron
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting single frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frames/{job_id}/{frame_index}")
async def get_frame(job_id: str, frame_index: int, wait: bool = True):
    """
    Get a specific frame from an extraction job

    Serves frames on-demand with polling support. If extraction is in progress,
    can optionally wait for the frame to become available.

    Args:
        job_id: Job identifier
        frame_index: Frame index to retrieve
        wait: If True, wait up to 30s for frame to be extracted

    Returns:
        FileResponse with frame image
    """
    try:
        from fastapi.responses import FileResponse
        import asyncio
        from pathlib import Path

        # Get job metadata
        metadata = await cache_manager.get_job_metadata(job_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        # Get job results to find frame path
        results = await cache_manager.get_job_results(job_id)

        # Check if extraction is complete
        if metadata.get("status") == JobStatus.COMPLETED.value and results:
            frames = results.get("frames", [])

            if frame_index < 0 or frame_index >= len(frames):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid frame_index {frame_index} (0-{len(frames)-1})"
                )

            frame_url = frames[frame_index]["url"]
            frame_path = frame_url.replace("file://", "")

            if not os.path.exists(frame_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame file not found: {frame_path}"
                )

            return FileResponse(
                frame_path,
                media_type="image/jpeg",
                headers={"X-Frame-Index": str(frame_index)}
            )

        # Extraction in progress - wait if requested
        if wait and metadata.get("status") in [JobStatus.QUEUED.value, JobStatus.PROCESSING.value]:
            max_wait = 30  # seconds
            poll_interval = 0.5  # seconds
            waited = 0

            while waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

                # Check if frame is ready
                updated_metadata = await cache_manager.get_job_metadata(job_id)
                if updated_metadata.get("status") == JobStatus.COMPLETED.value:
                    # Recursively call without wait to return the frame
                    return await get_frame(job_id, frame_index, wait=False)
                elif updated_metadata.get("status") == JobStatus.FAILED.value:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Job failed: {updated_metadata.get('error')}"
                    )

            # Timeout waiting
            raise HTTPException(
                status_code=408,
                detail=f"Timeout waiting for frame extraction (waited {max_wait}s)"
            )

        # Not waiting and not ready
        raise HTTPException(
            status_code=202,
            detail=f"Frame extraction in progress (status: {metadata.get('status')})"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frame: {e}", exc_info=True)
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

        # Check PyAV availability
        from .frame_extractor import PYAV_AVAILABLE

        methods = [
            "opencv_cuda" if gpu_available else "opencv_cpu",
            "ffmpeg",
            "sprites"
        ]
        if PYAV_AVAILABLE:
            methods.insert(1, "pyav_hw")
            methods.insert(2, "pyav_sw")

        return HealthResponse(
            status="healthy",
            service="frame-server",
            version="1.0.0",
            extraction_methods=methods,
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
    uvicorn.run(app, host="0.0.0.0", port=5001)
