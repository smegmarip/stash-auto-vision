"""
Faces Service - Main Application
FastAPI server for face recognition using InsightFace
"""

import os
import uuid
import time
import asyncio
import httpx
import cv2
import numpy as np
import magic
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from .models import (
    AnalyzeFacesRequest,
    AnalyzeJobResponse,
    AnalyzeJobStatus,
    AnalyzeJobResults,
    Face,
    Detection,
    BoundingBox,
    Landmarks,
    Demographics,
    Quality,
    QualityComponents,
    Occlusion,
    VideoMetadata,
    HealthResponse,
    JobStatus,
    ErrorResponse,
)
from .cache_manager import CacheManager
from .face_recognizer import FaceRecognizer
from .recognition_manager import RecognitionManager
from .frame_client import FrameServerClient
from .image_utils import normalize_image_if_needed, get_image_area

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_DEVICE = os.getenv("INSIGHTFACE_DEVICE", "cuda")
FRAME_SERVER_URL = os.getenv("FRAME_SERVER_URL", "http://frame-server:5001")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_ENHANCEMENT_PIXELS = int(os.getenv("MAX_ENHANCEMENT_PIXELS", "2073600"))  # 1920*1080

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instances
cache_manager: Optional[CacheManager] = None
face_recognizer: Optional[FaceRecognizer] = None
frame_client: Optional[FrameServerClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initialize services on startup, cleanup on shutdown
    """
    global cache_manager, face_recognizer, frame_client

    logger.info("Starting Faces Service...")

    # Initialize cache manager
    cache_manager = CacheManager(REDIS_URL, module="faces", ttl=CACHE_TTL)
    await cache_manager.connect()
    logger.info("Cache manager initialized")

    # Initialize recognition manager with multi-size support
    recognition_manager = RecognitionManager(model_name=INSIGHTFACE_MODEL, device=INSIGHTFACE_DEVICE)
    logger.info(f"Recognition manager initialized: {recognition_manager.get_model_info()}")

    # Initialize face recognizer with recognition manager
    face_recognizer = FaceRecognizer(
        model_name=INSIGHTFACE_MODEL,
        device=INSIGHTFACE_DEVICE,
        max_enhancement_pixels=MAX_ENHANCEMENT_PIXELS,
        recognition_manager=recognition_manager,
    )
    model_info = face_recognizer.get_model_info()
    logger.info(f"Face recognizer initialized: {model_info}")

    # Initialize frame-server client for enhancement
    frame_client = FrameServerClient(FRAME_SERVER_URL)

    yield

    # Cleanup
    logger.info("Shutting down Faces Service...")
    if cache_manager:
        await cache_manager.disconnect()
    if frame_client:
        await frame_client.close()
    logger.info("Faces Service stopped")


app = FastAPI(
    title="Faces Service", description="Face recognition service using InsightFace", version="1.0.0", lifespan=lifespan
)


def detect_source_type(source: str, source_type: Optional[str] = None) -> str:
    """
    Auto-detect source type from path/URL using MIME type detection

    Args:
        source: File path or URL
        source_type: Explicit type override (video, image, url)

    Returns:
        Detected source type: 'video', 'image', or 'url'
    """
    # URL detection
    if source.startswith("http://") or source.startswith("https://"):
        return "url"

    # File type detection using magic (MIME type)
    try:
        if os.path.exists(source):
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(source)

            if mime_type.startswith("video/"):
                return "video"
            elif mime_type.startswith("image/"):
                return "image"
            else:
                logger.warning(f"Unknown MIME type {mime_type} for {source}, falling back to extension")
        else:
            logger.warning(f"File not found: {source}, falling back to extension detection")
    except Exception as e:
        logger.warning(f"Error detecting MIME type: {e}, falling back to extension")

    # Use explicit source_type if provided
    if source_type:
        return source_type

    # Fallback to extension-based detection
    if source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v")):
        return "video"
    else:
        return "image"


async def get_video_info(video_path: str) -> tuple:
    """
    Get video information (duration, fps, total_frames) from OpenCV

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (duration_seconds, fps, total_frames)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        logger.info(f"Video info: {duration:.2f}s, {fps:.2f} FPS, {total_frames} frames")
        return duration, fps, total_frames
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        # Return safe defaults if can't read video
        return 60.0, 30.0, 1800


async def request_frames(video_path: str, job_id: str, parameters: dict) -> dict:
    """
    Request frame extraction from frame-server with adaptive sampling

    Automatically adjusts sampling_interval for short videos to ensure
    adequate frame coverage for face analysis.

    Args:
        video_path: Path to video file
        job_id: Job identifier
        parameters: Extraction parameters

    Returns:
        Frame extraction results
    """
    try:
        # Get video info to adjust sampling interval for short videos
        duration, fps, total_frames = await get_video_info(video_path)
        sampling_interval = parameters.get("sampling_interval", 2.0)

        # Adaptive sampling: ensure at least 20 frames for videos shorter than 20s
        # For videos <20s: interval = max(duration/20, 0.1)
        # For videos >=20s: use client-provided interval
        if duration < 20.0:
            adjusted_interval = max(duration / 20.0, 0.1)
            if adjusted_interval < sampling_interval:
                logger.info(
                    f"Short video ({duration:.1f}s): adjusting interval from {sampling_interval}s to {adjusted_interval:.2f}s"
                )
                sampling_interval = adjusted_interval

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Determine extraction method
            use_sprites = parameters.get("use_sprites", False)
            if use_sprites:
                extraction_method = "sprites"
            else:
                extraction_method = "opencv_cuda" if INSIGHTFACE_DEVICE == "cuda" else "opencv_cpu"

            # Submit extraction job
            response = await client.post(
                f"{FRAME_SERVER_URL}/extract",
                json={
                    "video_path": video_path,
                    "job_id": f"frames-{job_id}",
                    "extraction_method": extraction_method,
                    "sampling_strategy": {"mode": "interval", "interval_seconds": sampling_interval},
                    "scene_boundaries": parameters.get("scene_boundaries"),
                    "use_sprites": use_sprites,
                    "sprite_vtt_url": parameters.get("sprite_vtt_url"),
                    "sprite_image_url": parameters.get("sprite_image_url"),
                    "output_format": "jpeg",
                    "quality": 95,
                    "cache_duration": 3600,
                },
            )

            if response.status_code != 202:
                raise ValueError(f"Frame extraction failed: {response.text}")

            frame_job = response.json()
            frame_job_id = frame_job["job_id"]

            logger.info(f"Frame extraction job submitted: {frame_job_id}")

            # Poll for completion
            while True:
                status_response = await client.get(f"{FRAME_SERVER_URL}/jobs/{frame_job_id}/status")

                status = status_response.json()

                if status["status"] == "completed":
                    break
                elif status["status"] == "failed":
                    raise ValueError(f"Frame extraction failed: {status.get('error')}")

                await asyncio.sleep(2)

            # Get results
            results_response = await client.get(f"{FRAME_SERVER_URL}/jobs/{frame_job_id}/results")

            return results_response.json()

    except Exception as e:
        logger.error(f"Error requesting frames: {e}")
        raise


async def process_analysis_job(job_id: str, cache_key: str, request: AnalyzeFacesRequest):
    """
    Background task to process face analysis

    Args:
        job_id: Job identifier
        cache_key: Content-based cache key
        request: Analysis request parameters
    """
    try:
        logger.info(f"Starting job {job_id}")

        # Detect source type (video, image, or url) for metadata
        source_type = detect_source_type(request.source, request.source_type)
        logger.info(f"Detected source type: {source_type} for {request.source}")

        await cache_manager.update_job_status(
            job_id, status=JobStatus.PROCESSING.value, progress=0.0, stage="requesting_frames"
        )

        start_time = time.time()

        # Request frames from frame-server
        frame_results = await request_frames(request.source, job_id, request.parameters.dict())

        frames = frame_results["frames"]
        total_frames = len(frames)

        logger.info(f"Processing {total_frames} frames for job {job_id}")

        await cache_manager.update_job_status(
            job_id, status=JobStatus.PROCESSING.value, progress=0.1, stage="detecting_faces"
        )

        # Process each frame
        all_detections = []

        for idx, frame_info in enumerate(frames):
            # Load frame
            frame_url = frame_info["url"]
            frame_path = frame_url.replace("file://", "")

            frame = cv2.imread(frame_path)

            if frame is None:
                logger.warning(f"Failed to load frame: {frame_path}")
                continue

            # Detect faces (no enhancement yet - will enhance representative faces later)
            faces = await face_recognizer.detect_faces(
                frame, face_min_confidence=request.parameters.face_min_confidence
            )
            # Apply quality filtering with logging
            filtered_faces = []
            for f in faces:
                if f["quality"]["composite"] >= request.parameters.face_min_quality:
                    filtered_faces.append(f)
                else:
                    logger.debug(
                        f"Frame {idx}: Filtered face with quality={f['quality']['composite']:.3f} < "
                        f"min_quality={request.parameters.face_min_quality}"
                    )
            faces = filtered_faces

            logger.debug(f"Frame {idx}: {len(faces)} faces after detection and filtering")

            # Add frame metadata to detections
            for face in faces:
                detection = {**face, "frame_index": frame_info["index"], "timestamp": frame_info["timestamp"]}
                all_detections.append(detection)

            # Update progress every 5 frames or on the last frame
            progress = 0.1 + (0.7 * (idx + 1) / total_frames)
            logger.debug(
                f"Frame {idx + 1}/{total_frames}: progress={progress:.2f}, should_update={(idx + 1) % 5 == 0 or (idx + 1) == total_frames}"
            )
            if (idx + 1) % 5 == 0 or (idx + 1) == total_frames:
                logger.info(f"Updating progress: {idx + 1}/{total_frames} frames")
                await cache_manager.update_job_status(
                    job_id,
                    status=JobStatus.PROCESSING.value,
                    progress=progress,
                    stage="detecting_faces",
                    message=f"Processed {idx + 1}/{total_frames} frames",
                )

        logger.info(f"Found {len(all_detections)} total face detections")

        # Cluster faces
        await cache_manager.update_job_status(
            job_id, status=JobStatus.PROCESSING.value, progress=0.8, stage="clustering_faces"
        )

        if request.parameters.enable_deduplication:
            clusters = face_recognizer.cluster_faces(
                all_detections, similarity_threshold=request.parameters.embedding_similarity_threshold
            )
        else:
            # No clustering - each detection is a unique face
            clusters = {f"face_{i}": [i] for i in range(len(all_detections))}

        # Build face results
        faces = []

        for face_id, detection_indices in clusters.items():
            # Get representative detection
            rep_idx = face_recognizer.get_representative_detection(all_detections, detection_indices)

            rep_detection = all_detections[rep_idx]

            # Enhance representative face if needed
            if request.parameters.enhancement.enabled:
                quality = rep_detection["quality"]["composite"]
                confidence = rep_detection["confidence"]
                components = rep_detection["quality"]["components"]
                size_s = components["size"]
                sharpness_s = components["sharpness"]
                pose_s = components["pose"]
                image_area_pixels = get_image_area(frame)

                # Granular enhancement criteria:
                # - Size: small but viable (0.25-0.5 = ~100-165px)
                # - Sharpness: has detail to enhance (>= 0.25)
                # - Pose: feasible for enhancement (>= 0.5)
                granular_enhance = size_s >= 0.25 and size_s < 0.5 and sharpness_s >= 0.25 and pose_s >= 0.5

                # Optional composite override (still respects sharpness/pose gates)
                quality_trigger = request.parameters.enhancement.quality_trigger
                composite_override = (
                    quality_trigger > 0 and quality < quality_trigger and sharpness_s >= 0.25 and pose_s >= 0.5
                )

                # Final decision
                should_enhance = (
                    (granular_enhance or composite_override)
                    and quality >= request.parameters.face_min_quality
                    and confidence >= request.parameters.face_min_confidence
                    and not request.parameters.use_sprites
                    and image_area_pixels <= MAX_ENHANCEMENT_PIXELS
                )

                if should_enhance:
                    logger.info(
                        f"Enhancing representative face {face_id}: quality={quality:.3f}, confidence={confidence:.3f}"
                    )

                    # Load the frame for this detection
                    frame_idx = rep_detection["frame_index"]
                    frame_info = frames[frame_idx]
                    frame_path = frame_info["url"].replace("file://", "")
                    frame = cv2.imread(frame_path)

                    if frame is not None:
                        # Enhance frame
                        enhanced_frame_data = await frame_client.enhance_frame(
                            video_path=request.source,
                            timestamp=rep_detection["timestamp"],
                            model=request.parameters.enhancement.model,
                            fidelity_weight=request.parameters.enhancement.fidelity_weight,
                            output_format="jpeg",
                            quality=95,
                        )

                        if enhanced_frame_data:
                            # Decode enhanced frame
                            enhanced_image = cv2.imdecode(
                                np.frombuffer(enhanced_frame_data, np.uint8), cv2.IMREAD_COLOR
                            )

                            if enhanced_image is not None:
                                # Re-detect face in enhanced frame
                                enhanced_detections = await face_recognizer.detect_faces(
                                    enhanced_image, face_min_confidence=request.parameters.face_min_confidence
                                )

                                # Find best detection by quality
                                if enhanced_detections:
                                    best_enhanced = max(enhanced_detections, key=lambda d: d["quality"]["composite"])

                                    # Only use enhanced if quality improved
                                    if best_enhanced["quality"]["composite"] > quality:
                                        logger.info(
                                            f"Enhancement successful for {face_id}: "
                                            f"quality {quality:.3f} → {best_enhanced['quality']['composite']:.3f}"
                                        )
                                        # Update representative detection with enhanced version
                                        best_enhanced["enhanced"] = True
                                        best_enhanced["frame_index"] = rep_detection["frame_index"]
                                        best_enhanced["timestamp"] = rep_detection["timestamp"]
                                        rep_detection = best_enhanced
                                    else:
                                        logger.info(
                                            f"Enhancement did not improve quality for {face_id}: "
                                            f"{quality:.3f} → {best_enhanced['quality']['composite']:.3f}"
                                        )
                                else:
                                    logger.warning(f"No face detected in enhanced frame for {face_id}")
                            else:
                                logger.error(f"Failed to decode enhanced frame for {face_id}")
                        else:
                            logger.error(f"Failed to enhance frame for {face_id}")
                    else:
                        logger.warning(f"Failed to load frame for enhancement: {frame_path}")

            # Build detections list
            detections = [
                Detection(
                    frame_index=all_detections[i]["frame_index"],
                    timestamp=all_detections[i]["timestamp"],
                    bbox=BoundingBox(**all_detections[i]["bbox"]),
                    confidence=all_detections[i]["confidence"],
                    quality=Quality(
                        composite=all_detections[i]["quality"]["composite"],
                        components=QualityComponents(**all_detections[i]["quality"]["components"]),
                    ),
                    pose=all_detections[i]["pose"],
                    landmarks=Landmarks(**all_detections[i]["landmarks"]),
                    enhanced=all_detections[i].get("enhanced", False),
                    occlusion=Occlusion(**all_detections[i]["occlusion"]),
                )
                for i in detection_indices
            ]

            # Build face object
            face = Face(
                face_id=face_id,
                embedding=rep_detection["embedding"],
                demographics=(
                    Demographics(**rep_detection["demographics"]) if rep_detection.get("demographics") else None
                ),
                detections=detections,
                representative_detection=Detection(
                    frame_index=rep_detection["frame_index"],
                    timestamp=rep_detection["timestamp"],
                    bbox=BoundingBox(**rep_detection["bbox"]),
                    confidence=rep_detection["confidence"],
                    quality=Quality(
                        composite=rep_detection["quality"]["composite"],
                        components=QualityComponents(**rep_detection["quality"]["components"]),
                    ),
                    pose=rep_detection["pose"],
                    landmarks=Landmarks(**rep_detection["landmarks"]),
                    enhanced=rep_detection.get("enhanced", False),
                    occlusion=Occlusion(**rep_detection["occlusion"]),
                ),
            )

            faces.append(face)

        processing_time = time.time() - start_time

        # Build metadata (use original_source if provided, otherwise request.source)
        metadata = VideoMetadata(
            source=request.source,
            source_type=source_type,
            total_frames=frame_results["metadata"]["total_frames"],
            frames_processed=total_frames,
            unique_faces=len(faces),
            total_detections=len(all_detections),
            processing_time_seconds=processing_time,
            method=frame_results["metadata"]["extraction_method"],
            model=INSIGHTFACE_MODEL,
            frame_enhancement=request.parameters.enhancement if request.parameters.enhancement.enabled else None,
        )

        # Create results
        results = AnalyzeJobResults(
            job_id=job_id, source_id=request.source_id, status=JobStatus.COMPLETED, faces=faces, metadata=metadata
        )

        # Cache results
        await cache_manager.cache_job_results(
            job_id=job_id, cache_key=cache_key, results=results.dict(), ttl=request.parameters.cache_duration
        )

        # Update final status
        await cache_manager.update_job_status(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=1.0,
            stage="completed",
            message=f"Found {len(faces)} unique faces in {processing_time:.2f}s",
        )

        logger.info(f"Job {job_id} completed: {len(faces)} faces in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await cache_manager.update_job_status(job_id, status=JobStatus.FAILED.value, progress=0.0, error=str(e))


@app.post("/analyze", response_model=AnalyzeJobResponse, status_code=202)
async def analyze_faces(request: AnalyzeFacesRequest, background_tasks: BackgroundTasks):
    """Submit face analysis job"""
    try:
        if not os.path.exists(request.source):
            raise HTTPException(status_code=404, detail=f"Video not found: {request.source}")

        # Normalize image EXIF orientation if needed (for direct API calls)
        source_type = detect_source_type(request.source, request.source_type)
        if source_type == "image":
            job_id_for_norm = request.job_id or str(uuid.uuid4())
            normalized_path, was_normalized = normalize_image_if_needed(
                request.source, output_dir="/tmp/downloads", job_id=job_id_for_norm
            )
            if was_normalized:
                logger.info(f"Using EXIF-normalized image: {normalized_path}")
                request.source = normalized_path

        # Generate cache key
        params = request.parameters.dict()
        cache_key = cache_manager.generate_cache_key(request.source, params)

        # Check cache
        cached_job_id = await cache_manager.get_cached_job_id(cache_key)

        if cached_job_id:
            metadata = await cache_manager.get_job_metadata(cached_job_id)
            return AnalyzeJobResponse(
                job_id=cached_job_id,
                status=JobStatus(metadata.get("status", "completed")),
                created_at=metadata.get("created_at", ""),
            )

        # Create new job
        job_id = request.job_id or str(uuid.uuid4())

        metadata = {
            "job_id": job_id,
            "cache_key": cache_key,
            "status": JobStatus.QUEUED.value,
            "progress": 0.0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "source_id": request.source_id,
        }

        await cache_manager.cache_job_metadata(
            job_id=job_id, cache_key=cache_key, metadata=metadata, ttl=request.parameters.cache_duration
        )

        background_tasks.add_task(process_analysis_job, job_id, cache_key, request)

        logger.info(f"Job {job_id} queued")

        return AnalyzeJobResponse(job_id=job_id, status=JobStatus.QUEUED, created_at=metadata["created_at"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=AnalyzeJobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        result_summary = None
        if metadata.get("status") == JobStatus.COMPLETED.value:
            results = await cache_manager.get_job_results(job_id)
            if results:
                result_summary = {
                    "unique_faces": results["metadata"]["unique_faces"],
                    "total_detections": results["metadata"]["total_detections"],
                    "frames_processed": results["metadata"]["frames_processed"],
                    "processing_time_seconds": results["metadata"]["processing_time_seconds"],
                }

        return AnalyzeJobStatus(
            job_id=job_id,
            status=JobStatus(metadata["status"]),
            progress=metadata.get("progress", 0.0),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            created_at=metadata["created_at"],
            started_at=metadata.get("started_at"),
            completed_at=metadata.get("completed_at"),
            result_summary=result_summary,
            error=metadata.get("error"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/results", response_model=AnalyzeJobResults)
async def get_job_results(job_id: str):
    """Get job results"""
    try:
        metadata = await cache_manager.get_job_metadata(job_id)

        if not metadata:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        if metadata["status"] != JobStatus.COMPLETED.value:
            raise HTTPException(status_code=409, detail=f"Job not completed (status: {metadata['status']})")

        results = await cache_manager.get_job_results(job_id)

        if not results:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        return AnalyzeJobResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check"""
    try:
        cache_size_mb = await cache_manager.get_cache_size()
        active_jobs = await cache_manager.count_active_jobs()

        gpu_available = INSIGHTFACE_DEVICE == "cuda"

        return HealthResponse(
            status="healthy",
            service="faces-service",
            version="1.0.0",
            model=INSIGHTFACE_MODEL,
            gpu_available=gpu_available,
            active_jobs=active_jobs,
            cache_size_mb=cache_size_mb,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    import asyncio

    uvicorn.run(app, host="0.0.0.0", port=5003)
