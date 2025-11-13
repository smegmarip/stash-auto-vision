"""
Objects Service - Main Application (Stubbed)
Placeholder for YOLO-World-based object detection (Phase 3)
"""

import os
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Objects Service (Stubbed)",
    description="Object detection service - not yet implemented",
    version="1.0.0"
)


@app.post("/analyze", status_code=202)
async def analyze_objects(request: dict):
    """
    Stub endpoint for object analysis

    Phase 3 implementation will use YOLO-World for:
    - Open-vocabulary object detection
    - Custom object categories
    - Temporal object tracking
    - Bounding box annotations
    """
    logger.info("Received objects analysis request (not implemented)")

    return {
        "job_id": f"objects-stub-{int(time.time())}",
        "status": "not_implemented",
        "message": "Objects analysis module is not yet implemented (Phase 3)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    }


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Stub endpoint for job status"""
    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "status": "not_implemented",
            "message": "Objects analysis module is not yet implemented (Phase 3)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }
    )


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Stub endpoint for job results"""
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "Objects analysis module is not yet implemented (Phase 3)",
                "type": "NotImplementedError"
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "objects-service",
        "version": "1.0.0",
        "implemented": False,
        "phase": 3,
        "message": "Stub service - awaiting YOLO-World integration",
        "default_min_confidence": float(os.getenv("OBJECTS_MIN_CONFIDENCE", "0.5"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)
