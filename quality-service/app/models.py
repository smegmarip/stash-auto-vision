"""
Quality Service - Data Models
Pydantic models for request/response validation
"""

import numpy as np
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# Request/Response Models
class BBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Landmarks(BaseModel):
    left_eye: List[float]
    right_eye: List[float]
    nose: List[float]
    mouth_left: List[float]
    mouth_right: List[float]


class Pose(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class FaceData(BaseModel):
    bbox: BBox
    landmarks: Landmarks
    pose: Optional[Pose] = None


class OcclusionData(BaseModel):
    pred: int = Field(..., description="Predicted class (0=non-occluded, 1=occluded)")
    prob: float = Field(..., description="Confidence in predicted class (0.0-1.0)")


class QualityInput(BaseModel):
    face: FaceData
    occlusion: OcclusionData


class QualityAssessRequest(BaseModel):
    source: str = Field(..., description="Image source (URL, file path, or base64)")
    source_type: Literal["url", "path", "bytes"]
    input: QualityInput


class QualityComponents(BaseModel):
    size: float
    pose: float
    occlusion: float
    sharpness: float


class QualityScore(BaseModel):
    composite: float
    components: QualityComponents


class QualityAssessResponse(BaseModel):
    score: QualityScore


class HealthResponse(BaseModel):
    status: str
    model: str


# Mock InsightFace face object
class MockFace:
    """Mock InsightFace face object for FaceQuality.calculate_quality()"""

    def __init__(self, bbox, landmarks, pose):
        self.bbox = np.array([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max], dtype=np.float32)
        self.kps = np.array(
            [landmarks.left_eye, landmarks.right_eye, landmarks.nose, landmarks.mouth_left, landmarks.mouth_right],
            dtype=np.float32,
        )

        # Add pose if provided
        if pose:
            self.pose = [pose.pitch, pose.yaw, pose.roll]
