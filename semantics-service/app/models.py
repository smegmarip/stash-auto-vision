"""
Semantics Service - Data Models
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
    NOT_IMPLEMENTED = "not_implemented"


class SemanticsParameters(BaseModel):
    """Semantics analysis configuration"""

    model: str = Field(
        default=os.getenv("CLIP_MODEL", "google/siglip-base-patch16-224"),
        description="CLIP/SigLIP model variant"
    )
    classification_tags: Optional[List[str]] = Field(
        default=None,
        description="Predefined tags for zero-shot classification"
    )
    custom_prompts: Optional[List[str]] = Field(
        default=None,
        description="Custom text prompts for zero-shot classification"
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Generate multi-modal embeddings for similarity search"
    )
    min_confidence: float = Field(
        default=float(os.getenv("SEMANTICS_MIN_CONFIDENCE", "0.5")),
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for tag assignment"
    )
    top_k_tags: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of tags to return per frame"
    )
    batch_size: int = Field(
        default=int(os.getenv("SEMANTICS_BATCH_SIZE", "32")),
        ge=1,
        le=128,
        description="Batch size for inference"
    )
    sampling_interval: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Frame sampling interval in seconds"
    )
    scene_boundaries: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Scene boundaries from scenes-service (start_timestamp, end_timestamp)"
    )


class AnalyzeSemanticsRequest(BaseModel):
    """Request to analyze scene semantics"""

    source: str = Field(..., description="Video path or URL")
    source_id: str = Field(..., description="Scene identifier for reference")
    job_id: Optional[str] = Field(default=None, description="Parent job ID for tracking")
    frame_extraction_job_id: Optional[str] = Field(
        default=None,
        description="Job ID from Frame Server (reuse extracted frames)"
    )
    scenes_job_id: Optional[str] = Field(
        default=None,
        description="Job ID from Scenes Service (fetch pre-computed scene boundaries)"
    )
    parameters: SemanticsParameters = Field(default_factory=SemanticsParameters)


class SemanticTag(BaseModel):
    """Individual semantic tag with confidence"""

    tag: str = Field(..., description="Tag label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from CLIP/SigLIP")
    source: str = Field(
        ...,
        description="Tag source type: 'predefined', 'custom_prompt', or 'zero_shot'"
    )


class SceneClassification(BaseModel):
    """Hierarchical scene classification"""

    setting: Optional[str] = Field(None, description="Scene setting (indoor/outdoor)")
    location: Optional[str] = Field(None, description="Location type (kitchen, bedroom, etc.)")
    activity: Optional[str] = Field(None, description="Primary activity")


class FrameSemantics(BaseModel):
    """Per-frame semantic analysis results"""

    frame_index: int
    timestamp: float
    tags: List[SemanticTag]
    embedding: Optional[List[float]] = Field(
        None,
        description="Multi-modal embedding (512-D for SigLIP-B)"
    )
    scene_classification: Optional[SceneClassification] = None


class SceneSemanticSummary(BaseModel):
    """Aggregated semantic summary for a scene"""

    start_timestamp: float
    end_timestamp: float
    dominant_tags: List[str] = Field(description="Most frequent tags across frames")
    frame_count: int
    avg_confidence: float = Field(ge=0.0, le=1.0)


class SemanticsMetadata(BaseModel):
    """Processing metadata"""

    source: str
    source_type: Optional[str] = Field(default="video", description="Source type: 'video', 'image', or 'url'")
    total_frames: int
    model: str
    frames_analyzed: int
    processing_time_seconds: float
    device: str = Field(description="Device used: 'cuda' or 'cpu'")
    batch_size: int
    total_tags_generated: int


class SceneSemanticsOutcome(BaseModel):
    """Semantic analysis results for a scene"""

    frames: List[FrameSemantics]
    scene_summaries: Optional[List[SceneSemanticSummary]] = None
    metadata: SemanticsMetadata


class AnalyzeSemanticsResponse(BaseModel):
    """Response for semantics analysis job submission"""

    job_id: str
    status: JobStatus
    message: Optional[str] = None
    created_at: str
    cache_hit: bool = Field(default=False, description="Whether results were retrieved from cache")


class JobStatusResponse(BaseModel):
    """Job status response"""

    job_id: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

class SemanticsResults(BaseModel):
    """Complete semantics analysis results"""

    job_id: str
    source_id: str
    status: JobStatus
    semantics: SceneSemanticsOutcome
    metadata: SemanticsMetadata


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    service: str
    version: str
    implemented: bool
    phase: int
    message: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
    default_min_confidence: float


class Frame(BaseModel):
    """Extracted frame metadata"""

    index: int
    timestamp: float
    url: str
    width: int
    height: int

class FrameMetadata(BaseModel):
    """Metadata for a single extracted frame"""

    video_path: str
    extraction_method: str
    total_frames: int
    video_duration_seconds: float
    video_fps: float
    processing_time_seconds: float
    enhancement_enabled: bool = False
    enhancement_model: Optional[str] = None
    faces_enhanced: Optional[int] = None
    

class FramesExtractionResult(BaseModel):
    """Frame extraction result metadata"""

    job_id: str
    status: JobStatus
    cache_key: str
    frames: List[Frame]
    metadata: FrameMetadata