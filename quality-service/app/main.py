"""
Quality Service - Face Quality Assessment API
FastAPI service for comprehensive face quality scoring
"""

import os
import base64
import logging
import httpx
import cv2
import numpy as np
from typing import Optional, Dict, List, Literal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from .models import (
    MockFace,
    QualityAssessRequest,
    QualityComponents,
    QualityAssessResponse,
    QualityScore,
    HealthResponse,
)

from .face_quality import FaceQuality

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Environment configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MUSIQ_MODEL = os.getenv("MUSIQ_MODEL", "koniq-10k")

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instance
face_quality: Optional[FaceQuality] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FaceQuality on startup"""
    global face_quality

    logger.info("Starting Quality Service...")
    face_quality = FaceQuality()
    logger.info(f"Face quality assessment initialized with MUSIQ model: {MUSIQ_MODEL}")

    yield

    logger.info("Shutting down Quality Service...")


app = FastAPI(
    title="Quality Service",
    description="Face quality assessment with MUSIQ-based sharpness scoring",
    version="1.0.0",
    lifespan=lifespan,
)


async def load_image(source: str, source_type: str) -> np.ndarray:
    """Load image from source"""
    try:
        if source_type == "url":
            # Fetch from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(source)
                response.raise_for_status()
                image_bytes = response.content

            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        elif source_type == "path":
            # Load from file path
            if not Path(source).exists():
                raise FileNotFoundError(f"Image file not found: {source}")
            img = cv2.imread(source, cv2.IMREAD_COLOR)

        elif source_type == "bytes":
            # Decode base64
            image_bytes = base64.b64decode(source)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        else:
            raise ValueError(f"Invalid source_type: {source_type}")

        if img is None:
            raise ValueError("Failed to decode image")

        return img

    except Exception as e:
        logger.error(f"Error loading image from {source_type}: {e}")
        raise


@app.get("/quality/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(status="healthy", model=MUSIQ_MODEL)


@app.post("/quality/assess", response_model=QualityAssessResponse)
async def assess_quality(request: QualityAssessRequest):
    """
    Assess face quality with comprehensive scoring

    Accepts face data, occlusion info, and image source
    Returns composite quality score + component breakdown
    """
    try:
        # Load image
        img = await load_image(request.source, request.source_type)
        logger.debug(f"Loaded image: {img.shape}")

        # Create mock InsightFace face object
        mock_face = MockFace(
            bbox=request.input.face.bbox, landmarks=request.input.face.landmarks, pose=request.input.face.pose
        )

        # Calculate quality using FaceQuality
        composite_score = face_quality.calculate_quality(
            face=mock_face, frame=img, occlusion_data=(request.input.occlusion.pred, request.input.occlusion.prob)
        )

        # Get component scores (re-calculate to extract individual values)
        bbox = request.input.face.bbox
        ih, iw = img.shape[0:2]
        x1, y1 = max(0, bbox.x_min), max(0, bbox.y_min)
        x2, y2 = min(iw, bbox.x_max), min(ih, bbox.y_max)
        h, w = y2 - y1, x2 - x1
        face_min_dim = min(h, w)

        # Individual component scores
        size_score = face_quality._size_score(face_min_dim)

        yaw, pitch, _ = face_quality._estimate_pose_angles(mock_face)
        pose_score = face_quality._pose_score(yaw, pitch)

        occlusion_score = face_quality._occlusion_score(request.input.occlusion.pred, request.input.occlusion.prob)

        face_img = img[y1:y2, x1:x2]
        sharpness_score = face_quality._sharpness_score(face_img) if face_img.size else 0.25

        return QualityAssessResponse(
            score=QualityScore(
                composite=composite_score,
                components=QualityComponents(
                    size=size_score, pose=pose_score, occlusion=occlusion_score, sharpness=sharpness_score
                ),
            )
        )

    except Exception as e:
        logger.error(f"Error assessing quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5006)
