"""
Vision API - Main Application
Orchestrator for all vision analysis services
"""

import os
import uuid
import time
import httpx
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import redis.asyncio as aioredis

from .models import (
    AnalyzeVideoRequest,
    AnalyzeJobResponse,
    AnalyzeJobStatus,
    AnalyzeJobResults,
    ServiceJobInfo,
    HealthResponse
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
SCENES_SERVICE_URL = os.getenv("SCENES_SERVICE_URL", "http://scenes-service:5002")
FACES_SERVICE_URL = os.getenv("FACES_SERVICE_URL", "http://faces-service:5003")
SEMANTICS_SERVICE_URL = os.getenv("SEMANTICS_SERVICE_URL", "http://semantics-service:5004")
OBJECTS_SERVICE_URL = os.getenv("OBJECTS_SERVICE_URL", "http://objects-service:5005")

redis_client: Optional[aioredis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client

    logger.info("Starting Vision API Orchestrator...")

    # Connect to Redis
    redis_client = await aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    logger.info("Connected to Redis")

    yield

    # Cleanup
    logger.info("Shutting down Vision API...")
    if redis_client:
        await redis_client.close()
    logger.info("Vision API stopped")


app = FastAPI(
    title="Vision API",
    description="Orchestrator for video vision analysis services",
    version="1.0.0",
    lifespan=lifespan
)


async def call_service(
    service_name: str,
    service_url: str,
    request_data: dict,
    orchestrator_job_id: Optional[str] = None,
    orchestrator_metadata: Optional[Dict[str, Any]] = None,
    base_progress: float = 0.0,
    progress_weight: float = 0.25,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Call a vision service and wait for results

    Args:
        service_name: Service name for logging
        service_url: Base URL of service
        request_data: Request payload
        orchestrator_job_id: Parent orchestrator job ID for progress updates
        orchestrator_metadata: Parent metadata dict to update
        base_progress: Starting progress value for this service (0.0-1.0)
        progress_weight: Progress range this service represents (e.g., 0.25 = 25%)
        timeout: Timeout in seconds

    Returns:
        Service results
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Submit job
            logger.info(f"Submitting job to {service_name}...")

            if service_name == "scenes":
                endpoint = f"{service_url}/detect"
            else:
                endpoint = f"{service_url}/analyze"

            response = await client.post(endpoint, json=request_data)

            if response.status_code != 202:
                raise ValueError(f"{service_name} job submission failed: {response.text}")

            job_info = response.json()
            job_id = job_info["job_id"]

            logger.info(f"{service_name} job submitted: {job_id}")

            # Poll for completion
            while True:
                status_response = await client.get(
                    f"{service_url}/jobs/{job_id}/status"
                )

                status = status_response.json()

                # Update orchestrator progress based on sub-service progress
                if orchestrator_job_id and orchestrator_metadata:
                    sub_progress = status.get("progress", 0.0)
                    # Calculate overall progress: base + (sub_progress * weight)
                    overall_progress = base_progress + (sub_progress * progress_weight)

                    sub_stage = status.get("stage", "")
                    sub_message = status.get("message", "")

                    await update_metadata(
                        orchestrator_job_id,
                        orchestrator_metadata,
                        progress=overall_progress,
                        message=f"{service_name}: {sub_message}" if sub_message else f"Processing {service_name}..."
                    )
                    logger.debug(f"{service_name} progress: {sub_progress:.1%} -> overall: {overall_progress:.1%}")

                if status["status"] == "completed":
                    break
                elif status["status"] == "failed":
                    raise ValueError(f"{service_name} job failed: {status.get('error')}")
                elif status["status"] == "not_implemented":
                    logger.info(f"{service_name} not implemented (stub)")
                    return {"status": "not_implemented", "job_id": job_id}

                await asyncio.sleep(2)

            # Get results
            results_response = await client.get(
                f"{service_url}/jobs/{job_id}/results"
            )

            return results_response.json()

    except Exception as e:
        logger.error(f"Error calling {service_name}: {e}")
        raise


async def update_metadata(
    job_id: str,
    metadata: Dict[str, Any],
    status: Optional[str] = None,
    progress: Optional[float] = None,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    error: Optional[str] = None
):
    """
    Helper to update job metadata in Redis

    Args:
        job_id: Job identifier
        metadata: Current metadata dict
        status: Job status
        progress: Progress fraction (0.0-1.0)
        stage: Current processing stage
        message: Status message
        error: Error message if failed
    """
    if status is not None:
        metadata["status"] = status
    if progress is not None:
        metadata["progress"] = progress
    if stage is not None:
        metadata["stage"] = stage
    if message is not None:
        metadata["message"] = message
    if error is not None:
        metadata["error"] = error

    await redis_client.setex(
        f"vision:job:{job_id}:metadata",
        3600,
        str(metadata)
    )


async def process_video_analysis(
    job_id: str,
    request: AnalyzeVideoRequest
):
    """
    Background task to orchestrate video analysis

    Args:
        job_id: Job identifier
        request: Analysis request
    """
    try:
        logger.info(f"Starting orchestrated analysis job {job_id}")

        start_time = time.time()

        # Store job metadata
        metadata = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "stage": "initializing",
            "message": "Starting video analysis",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "scene_id": request.scene_id,
            "video_path": request.source,
            "processing_mode": request.processing_mode
        }

        await redis_client.setex(
            f"vision:job:{job_id}:metadata",
            3600,
            str(metadata)
        )

        results = {
            "scenes": None,
            "faces": None,
            "semantics": None,
            "objects": None
        }

        # Sequential processing (default)
        if request.processing_mode == "sequential":
            # Calculate progress weights based on enabled services
            enabled_count = sum([
                request.modules.scenes.enabled,
                request.modules.faces.enabled,
                request.modules.semantics.enabled,
                request.modules.objects.enabled
            ])
            progress_per_service = 1.0 / enabled_count if enabled_count > 0 else 0.25
            progress = 0.0

            # Step 1: Scene detection
            if request.modules.scenes.enabled:
                logger.info("Running scene detection...")
                await update_metadata(
                    job_id, metadata,
                    progress=progress,
                    stage="scene_detection",
                    message="Detecting scene boundaries"
                )

                scenes_request = {
                    "video_path": request.source,
                    "job_id": f"scenes-{job_id}",
                    "detection_method": request.modules.scenes.parameters.get("scene_detection_method", "content"),
                    "scene_threshold": request.modules.scenes.parameters.get("scene_threshold", float(os.getenv("SCENES_THRESHOLD", "27.0"))),
                    "min_scene_length": request.modules.scenes.parameters.get("min_scene_length", 0.6)
                }

                scenes_result = await call_service(
                    "scenes",
                    SCENES_SERVICE_URL,
                    scenes_request,
                    orchestrator_job_id=job_id,
                    orchestrator_metadata=metadata,
                    base_progress=progress,
                    progress_weight=progress_per_service
                )
                results["scenes"] = scenes_result
                # Update local progress tracker (call_service already updated Redis)
                progress += progress_per_service
                # Update completion message only (progress already at correct value from call_service)
                await update_metadata(
                    job_id, metadata,
                    message=f"Scene detection complete ({len(scenes_result.get('scenes', []))} scenes found)"
                )

            # Step 2: Face recognition
            if request.modules.faces.enabled:
                logger.info("Running face recognition...")
                # Don't update progress here - it's already set from previous service
                await update_metadata(
                    job_id, metadata,
                    stage="face_recognition",
                    message="Analyzing faces in video"
                )

                # Pass scene boundaries if available
                scene_boundaries = None
                if results["scenes"] and results["scenes"].get("scenes"):
                    scene_boundaries = [
                        {
                            "start_timestamp": scene["start_timestamp"],
                            "end_timestamp": scene["end_timestamp"]
                        }
                        for scene in results["scenes"]["scenes"]
                    ]

                faces_request = {
                    "video_path": request.source,
                    "scene_id": request.scene_id,
                    "job_id": f"faces-{job_id}",
                    "parameters": {
                        "face_min_confidence": request.modules.faces.parameters.get("face_min_confidence", float(os.getenv("FACES_MIN_CONFIDENCE", "0.9"))),
                        "max_faces": request.modules.faces.parameters.get("max_faces", 50),
                        "sampling_interval": request.modules.faces.parameters.get("face_sampling_interval", 2.0),
                        "enable_deduplication": request.modules.faces.parameters.get("enable_deduplication", True),
                        "embedding_similarity_threshold": request.modules.faces.parameters.get("similarity_threshold", 0.6),
                        "detect_demographics": request.modules.faces.parameters.get("detect_demographics", True),
                        "scene_boundaries": scene_boundaries
                    }
                }

                faces_result = await call_service(
                    "faces",
                    FACES_SERVICE_URL,
                    faces_request,
                    orchestrator_job_id=job_id,
                    orchestrator_metadata=metadata,
                    base_progress=progress,
                    progress_weight=progress_per_service
                )
                results["faces"] = faces_result
                # Update local progress tracker
                progress += progress_per_service

                face_count = len(faces_result.get("faces", []))
                # Update completion message only (progress already at correct value)
                await update_metadata(
                    job_id, metadata,
                    message=f"Face recognition complete ({face_count} unique faces found)"
                )

            # Step 3: Semantics (stub)
            if request.modules.semantics.enabled:
                logger.info("Running semantics analysis (stubbed)...")
                await update_metadata(
                    job_id, metadata,
                    stage="semantic_analysis",
                    message="Analyzing scene semantics"
                )

                semantics_request = {
                    "video_path": request.source,
                    "scene_id": request.scene_id,
                    "parameters": {
                        "semantics_min_confidence": request.modules.semantics.parameters.get("semantics_min_confidence", float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.5")))
                    }
                }

                semantics_result = await call_service(
                    "semantics",
                    SEMANTICS_SERVICE_URL,
                    semantics_request,
                    orchestrator_job_id=job_id,
                    orchestrator_metadata=metadata,
                    base_progress=progress,
                    progress_weight=progress_per_service
                )
                results["semantics"] = semantics_result
                # Update local progress tracker
                progress += progress_per_service
                # Update completion message only
                await update_metadata(
                    job_id, metadata,
                    message="Semantic analysis complete"
                )

            # Step 4: Objects (stub)
            if request.modules.objects.enabled:
                logger.info("Running object detection (stubbed)...")
                await update_metadata(
                    job_id, metadata,
                    stage="object_detection",
                    message="Detecting objects in video"
                )

                objects_request = {
                    "video_path": request.source,
                    "scene_id": request.scene_id,
                    "parameters": {
                        "objects_min_confidence": request.modules.objects.parameters.get("objects_min_confidence", float(os.getenv("OBJECTS_MIN_CONFIDENCE", "0.5")))
                    }
                }

                objects_result = await call_service(
                    "objects",
                    OBJECTS_SERVICE_URL,
                    objects_request,
                    orchestrator_job_id=job_id,
                    orchestrator_metadata=metadata,
                    base_progress=progress,
                    progress_weight=progress_per_service
                )
                results["objects"] = objects_result
                # Update local progress tracker
                progress += progress_per_service
                # Update completion message only
                await update_metadata(
                    job_id, metadata,
                    message="Object detection complete"
                )

        # Parallel processing (future optimization)
        else:
            tasks = []

            if request.modules.scenes.enabled:
                scenes_request = {
                    "video_path": request.source,
                    "job_id": f"scenes-{job_id}"
                }
                tasks.append(call_service("scenes", SCENES_SERVICE_URL, scenes_request))

            if request.modules.faces.enabled:
                faces_request = {
                    "video_path": request.source,
                    "scene_id": request.scene_id,
                    "job_id": f"faces-{job_id}"
                }
                tasks.append(call_service("faces", FACES_SERVICE_URL, faces_request))

            # Wait for all tasks
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for idx, result in enumerate(task_results):
                if isinstance(result, Exception):
                    logger.error(f"Task {idx} failed: {result}")
                else:
                    if idx == 0 and request.modules.scenes.enabled:
                        results["scenes"] = result
                    elif idx == 1 and request.modules.faces.enabled:
                        results["faces"] = result

        processing_time = time.time() - start_time

        # Build summary for result_summary field
        result_summary = {}
        if results["scenes"]:
            result_summary["scenes"] = len(results["scenes"].get("scenes", []))
        if results["faces"]:
            result_summary["faces"] = len(results["faces"].get("faces", []))
        if results["semantics"]:
            result_summary["semantics"] = "completed"
        if results["objects"]:
            result_summary["objects"] = "completed"

        # Build final results
        final_results = {
            "job_id": job_id,
            "scene_id": request.scene_id,
            "status": "completed",
            **results,
            "metadata": {
                "processing_time_seconds": processing_time,
                "processing_mode": request.processing_mode,
                "services_used": {
                    "scenes": request.modules.scenes.enabled,
                    "faces": request.modules.faces.enabled,
                    "semantics": request.modules.semantics.enabled,
                    "objects": request.modules.objects.enabled
                }
            }
        }

        # Cache results
        await redis_client.setex(
            f"vision:job:{job_id}:results",
            3600,
            str(final_results)
        )

        # Update metadata with completion status
        await update_metadata(
            job_id, metadata,
            status="completed",
            progress=1.0,
            stage="completed",
            message=f"Analysis complete in {processing_time:.1f}s"
        )
        metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        metadata["result_summary"] = result_summary

        await redis_client.setex(
            f"vision:job:{job_id}:metadata",
            3600,
            str(metadata)
        )

        logger.info(f"Job {job_id} completed in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)

        await update_metadata(
            job_id, metadata,
            status="failed",
            stage="failed",
            message=f"Job failed: {str(e)[:100]}",
            error=str(e)
        )
        metadata["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

        await redis_client.setex(
            f"vision:job:{job_id}:metadata",
            3600,
            str(metadata)
        )


@app.post("/vision/analyze", response_model=AnalyzeJobResponse, status_code=202)
async def analyze_video(
    request: AnalyzeVideoRequest,
    background_tasks: BackgroundTasks
):
    """Submit comprehensive video analysis job"""
    try:
        if not os.path.exists(request.source):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {request.source}"
            )

        job_id = request.job_id or str(uuid.uuid4())

        # Queue background task
        background_tasks.add_task(process_video_analysis, job_id, request)

        logger.info(f"Orchestrated job {job_id} queued")

        return AnalyzeJobResponse(
            job_id=job_id,
            status="queued",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            services_enabled={
                "scenes": request.modules.scenes.enabled,
                "faces": request.modules.faces.enabled,
                "semantics": request.modules.semantics.enabled,
                "objects": request.modules.objects.enabled
            },
            processing_mode=request.processing_mode
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vision/jobs/{job_id}/status", response_model=AnalyzeJobStatus)
async def get_job_status(job_id: str):
    """Get orchestrated job status"""
    try:
        metadata_str = await redis_client.get(f"vision:job:{job_id}:metadata")

        if not metadata_str:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        metadata = eval(metadata_str)

        return AnalyzeJobStatus(
            job_id=job_id,
            status=metadata["status"],
            progress=metadata.get("progress", 0.0),
            processing_mode=metadata.get("processing_mode", "sequential"),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            services=[],  # Could expand to show individual service statuses
            created_at=metadata["created_at"],
            started_at=metadata.get("started_at"),
            completed_at=metadata.get("completed_at"),
            result_summary=metadata.get("result_summary"),
            error=metadata.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vision/jobs/{job_id}/results", response_model=AnalyzeJobResults)
async def get_job_results(job_id: str):
    """Get orchestrated job results"""
    try:
        metadata_str = await redis_client.get(f"vision:job:{job_id}:metadata")

        if not metadata_str:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        metadata = eval(metadata_str)

        if metadata["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job not completed (status: {metadata['status']})"
            )

        results_str = await redis_client.get(f"vision:job:{job_id}:results")

        if not results_str:
            raise HTTPException(status_code=404, detail=f"Results not found for job: {job_id}")

        results = eval(results_str)

        return AnalyzeJobResults(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for orchestrator and all services"""
    try:
        services = {}

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check each service
            for name, url in [
                ("scenes", SCENES_SERVICE_URL),
                ("faces", FACES_SERVICE_URL),
                ("semantics", SEMANTICS_SERVICE_URL),
                ("objects", OBJECTS_SERVICE_URL)
            ]:
                try:
                    response = await client.get(f"{url}/health")
                    services[name] = response.json()
                except Exception as e:
                    services[name] = {"status": "unhealthy", "error": str(e)}

        return HealthResponse(
            status="healthy",
            service="vision-api",
            version="1.0.0",
            services=services
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
