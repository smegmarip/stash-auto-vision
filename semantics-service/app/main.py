"""
Semantics Service - Main Application (Stubbed)
Placeholder for CLIP-based scene classification (Phase 2)
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
    title="Semantics Service (Stubbed)",
    description="Scene classification service - not yet implemented",
    version="1.0.0"
)


@app.post("/analyze", status_code=202)
async def analyze_semantics(request: dict):
    """
    Stub endpoint for semantics analysis

    Phase 2 implementation will use CLIP for:
    - Scene classification (indoor/outdoor, setting types)
    - Action recognition
    - Custom text prompts
    - Multi-modal embeddings
    """
    logger.info("Received semantics analysis request (not implemented)")

    return {
        "job_id": f"semantics-stub-{int(time.time())}",
        "status": "not_implemented",
        "message": "Semantics analysis module is not yet implemented (Phase 2)",
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
            "message": "Semantics analysis module is not yet implemented (Phase 2)",
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
                "message": "Semantics analysis module is not yet implemented (Phase 2)",
                "type": "NotImplementedError"
            }
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "semantics-service",
        "version": "1.0.0",
        "implemented": False,
        "phase": 2,
        "message": "Stub service - awaiting CLIP integration",
        "default_min_confidence": float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.5"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)
