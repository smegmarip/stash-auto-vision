"""
Scenes Service - Data Models
Pydantic models for request/response validation
"""

import os
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DetectionMethod(str, Enum):
    """Scene detection methods"""
    CONTENT_DETECTOR = "content"  # ContentDetector (default)
    THRESHOLD_DETECTOR = "threshold"  # ThresholdDetector
    ADAPTIVE_DETECTOR = "adaptive"  # AdaptiveDetector


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectScenesRequest(BaseModel):
    """Request to detect scene boundaries in video"""
    video_path: str = Field(..., description="Absolute path to video file")
    job_id: Optional[str] = Field(default=None, description="Parent job ID for tracking")
    detection_method: DetectionMethod = DetectionMethod.CONTENT_DETECTOR
    scene_threshold: float = Field(default=float(os.getenv("SCENES_THRESHOLD", "27.0")), ge=0.0, le=100.0, description="Detection threshold")
    min_scene_length: float = Field(default=0.6, ge=0.1, description="Minimum scene length in seconds")
    cache_duration: int = Field(default=int(os.getenv("CACHE_TTL", "31536000")), ge=0, description="Cache TTL in seconds")


class SceneBoundary(BaseModel):
    """Scene boundary information"""
    scene_number: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    duration: float


class DetectJobResponse(BaseModel):
    """Response for scene detection job submission"""
    job_id: str
    status: JobStatus
    created_at: str
    cache_key: str
    estimated_scenes: Optional[int] = None


class DetectJobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    scenes_detected: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class VideoMetadata(BaseModel):
    """Video file metadata"""
    video_path: str
    detection_method: str
    total_frames: int
    video_duration_seconds: float
    video_fps: float
    processing_time_seconds: float


class DetectJobResults(BaseModel):
    """Complete job results"""
    job_id: str
    status: JobStatus
    cache_key: str
    scenes: List[SceneBoundary]
    metadata: VideoMetadata


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "scenes-service"
    version: str = "1.0.0"
    detection_methods: List[str]
    gpu_available: bool
    active_jobs: int
    cache_size_mb: float


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any]
