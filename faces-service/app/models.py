"""
Faces Service - Data Models
Pydantic models for request/response validation
"""

import os
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EnhancementParameters(BaseModel):
    """Face enhancement configuration"""
    enabled: bool = Field(default=False, description="Enable face enhancement")
    quality_trigger: float = Field(default=float(os.getenv("FACES_ENHANCEMENT_QUALITY_TRIGGER", "0.5")), ge=0.0, le=1.0, description="Trigger enhancement if quality below this")
    model: str = Field(default="codeformer", description="Enhancement model: 'gfpgan' or 'codeformer'")
    fidelity_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Fidelity vs quality tradeoff")


class FaceAnalysisParameters(BaseModel):
    """Face analysis configuration"""
    face_min_confidence: float = Field(default=float(os.getenv("FACES_MIN_CONFIDENCE", "0.9")), ge=0.0, le=1.0)
    face_min_quality: float = Field(default=float(os.getenv("FACES_MIN_QUALITY", "0.0")), ge=0.0, le=1.0, description="Minimum quality threshold (filter below this)")
    max_faces: int = Field(default=50, ge=1, le=1000)
    sampling_interval: float = Field(default=2.0, ge=0.1, le=10.0, description="Sampling interval in seconds (auto-adjusted for short videos)")
    use_sprites: bool = Field(default=False)
    sprite_vtt_url: Optional[str] = None
    sprite_image_url: Optional[str] = None
    enable_deduplication: bool = Field(default=True)
    embedding_similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    detect_demographics: bool = Field(default=True)
    scene_boundaries: Optional[List[Dict[str, float]]] = None
    cache_duration: int = Field(default=3600, ge=0)
    enhancement: EnhancementParameters = Field(default_factory=EnhancementParameters)


class AnalyzeFacesRequest(BaseModel):
    """Request to analyze faces in video/image"""
    source: str = Field(..., description="Path, URL, or image source to analyze")
    source_type: Optional[str] = Field(default=None, description="Source type: 'video', 'image', 'url' (auto-detected if omitted)")
    scene_id: str = Field(..., description="Scene ID for reference")
    job_id: Optional[str] = Field(default=None, description="Parent job ID for tracking")
    parameters: FaceAnalysisParameters = Field(default_factory=FaceAnalysisParameters)


class BoundingBox(BaseModel):
    """Face bounding box coordinates"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Landmarks(BaseModel):
    """Facial landmarks (5-point)"""
    left_eye: List[int]
    right_eye: List[int]
    nose: List[int]
    mouth_left: List[int]
    mouth_right: List[int]


class Demographics(BaseModel):
    """Demographic information"""
    age: int
    gender: str  # "M" or "F"
    emotion: str  # "neutral", "happy", "sad", "angry", "surprise", "disgust", "fear"


class Detection(BaseModel):
    """Single face detection"""
    frame_index: int
    timestamp: float
    bbox: BoundingBox
    confidence: float
    quality_score: float
    pose: str  # "front", "left", "right", "front-rotate-left", "front-rotate-right"
    landmarks: Landmarks
    enhanced: bool = False  # Indicates if face was enhanced via CodeFormer/GFPGAN
    occluded: bool = False  # Indicates if face is occluded (glasses, mask, hand, etc.)
    occlusion_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability that face is occluded (0.0-1.0)")


class Face(BaseModel):
    """Unique face cluster"""
    face_id: str
    embedding: List[float]  # 512-D ArcFace embedding
    demographics: Optional[Demographics] = None
    detections: List[Detection]
    representative_detection: Detection


class AnalyzeJobResponse(BaseModel):
    """Response for face analysis job submission"""
    job_id: str
    status: JobStatus
    created_at: str


class AnalyzeJobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VideoMetadata(BaseModel):
    """Video file metadata"""
    source: str
    total_frames: int
    frames_processed: int
    unique_faces: int
    total_detections: int
    processing_time_seconds: float
    method: str
    model: str
    frame_enhancement: Optional[EnhancementParameters] = Field(
        default=None,
        description="Frame enhancement settings used (if enhancement was enabled)"
    )


class AnalyzeJobResults(BaseModel):
    """Complete job results"""
    job_id: str
    scene_id: str
    status: JobStatus
    faces: List[Face]
    metadata: VideoMetadata


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "faces-service"
    version: str = "1.0.0"
    model: str
    gpu_available: bool
    active_jobs: int
    cache_size_mb: float


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any]
