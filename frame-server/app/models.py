"""
Frame Server - Data Models
Pydantic models for request/response validation
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    """Frame extraction methods"""
    OPENCV_CUDA = "opencv_cuda"
    OPENCV_CPU = "opencv_cpu"
    FFMPEG = "ffmpeg"


class SamplingMode(str, Enum):
    """Frame sampling strategies"""
    INTERVAL = "interval"
    TIMESTAMPS = "timestamps"
    SCENE_BASED = "scene_based"


class OutputFormat(str, Enum):
    """Output image format"""
    JPEG = "jpeg"
    PNG = "png"


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SamplingStrategy(BaseModel):
    """Frame sampling configuration"""
    mode: SamplingMode = SamplingMode.INTERVAL
    interval_seconds: Optional[float] = Field(default=2.0, ge=0.1, le=10.0)
    timestamps: Optional[List[float]] = None


class ExtractFramesRequest(BaseModel):
    """Request to extract frames from video"""
    video_path: str = Field(..., description="Absolute path to video file")
    job_id: Optional[str] = Field(default=None, description="Parent job ID for tracking")
    extraction_method: ExtractionMethod = ExtractionMethod.OPENCV_CUDA
    sampling_strategy: SamplingStrategy = Field(default_factory=SamplingStrategy)
    use_sprites: bool = Field(default=False, description="Use sprite sheets if available")
    sprite_vtt_url: Optional[str] = None
    sprite_image_url: Optional[str] = None
    scene_boundaries: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Scene boundaries from scenes-service"
    )
    output_format: OutputFormat = OutputFormat.JPEG
    quality: int = Field(default=95, ge=1, le=100, description="JPEG quality")
    cache_duration: int = Field(default=3600, ge=0, description="Cache TTL in seconds")


class FrameMetadata(BaseModel):
    """Metadata for a single extracted frame"""
    index: int
    timestamp: float
    url: str
    width: int
    height: int


class ExtractJobResponse(BaseModel):
    """Response for frame extraction job submission"""
    job_id: str
    status: JobStatus
    created_at: str
    cache_key: str
    estimated_frames: Optional[int] = None


class ExtractJobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    frames_extracted: Optional[int] = None
    frames_total: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class VideoMetadata(BaseModel):
    """Video file metadata"""
    video_path: str
    extraction_method: str
    total_frames: int
    video_duration_seconds: float
    video_fps: float
    processing_time_seconds: float


class ExtractJobResults(BaseModel):
    """Complete job results"""
    job_id: str
    status: JobStatus
    cache_key: str
    frames: List[FrameMetadata]
    metadata: VideoMetadata


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "frame-server"
    version: str = "1.0.0"
    extraction_methods: List[str]
    gpu_available: bool
    active_jobs: int
    cache_size_mb: float


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any]
