"""
Captioning Service - Pydantic Models
Request/response schemas for JoyCaption VLM integration
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_IMPLEMENTED = "not_implemented"
    WAITING_FOR_GPU = "waiting_for_gpu"


class PromptType(str, Enum):
    """JoyCaption prompt types"""
    DESCRIPTIVE = "descriptive"
    DESCRIPTIVE_INFORMAL = "descriptive_informal"
    STRAIGHTFORWARD = "straightforward"
    BOORU_LIKE = "booru_like"
    BOORU_LIKE_EXTENDED = "booru_like_extended"
    ART_CRITIC = "art_critic"
    TRAINING_PROMPT = "training_prompt"
    MLP_TAGS = "mlp_tags"


class FrameSelectionMethod(str, Enum):
    """Frame selection strategies"""
    SCENE_BASED = "scene_based"      # N frames per scene boundary
    INTERVAL = "interval"             # Every N seconds
    SPRITE_SHEET = "sprite_sheet"     # Use existing sprites
    KEYFRAMES = "keyframes"           # Extract keyframes only


class CaptionParameters(BaseModel):
    """Parameters for captioning analysis"""
    prompt_type: PromptType = Field(
        default=PromptType.BOORU_LIKE,
        description="Primary prompt type for JoyCaption"
    )
    fallback_prompt_type: Optional[PromptType] = Field(
        default=PromptType.STRAIGHTFORWARD,
        description="Fallback prompt type if primary yields poor results"
    )
    frame_selection: FrameSelectionMethod = Field(
        default=FrameSelectionMethod.SCENE_BASED,
        description="How to select frames for captioning"
    )
    frames_per_scene: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Frames to caption per scene (scene_based mode)"
    )
    sampling_interval: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Interval in seconds (interval mode)"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for tag alignment"
    )
    max_tags_per_frame: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum tags to extract per frame"
    )
    align_to_taxonomy: bool = Field(
        default=True,
        description="Align VLM output to Stash tag taxonomy"
    )
    generate_embeddings: bool = Field(
        default=False,
        description="Generate text embeddings for captions"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Batch size for inference (limited by VRAM)"
    )
    use_quantization: bool = Field(
        default=True,
        description="Use 4-bit quantization to reduce VRAM"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt override (advanced)"
    )


class AnalyzeCaptionsRequest(BaseModel):
    """Request to analyze video with JoyCaption"""
    source: str = Field(
        ...,
        description="Path to video file"
    )
    source_id: str = Field(
        ...,
        description="Unique identifier for the source (e.g., Stash scene ID)"
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Optional job ID (generated if not provided)"
    )
    scenes_job_id: Optional[str] = Field(
        default=None,
        description="Existing scenes job ID to use for scene boundaries"
    )
    parameters: CaptionParameters = Field(
        default_factory=CaptionParameters,
        description="Captioning parameters"
    )


class AnalyzeCaptionsResponse(BaseModel):
    """Response for caption analysis request"""
    job_id: str
    status: JobStatus
    message: Optional[str] = None
    created_at: str
    cache_hit: bool = False


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    gpu_wait_position: Optional[int] = None


class CaptionTag(BaseModel):
    """A single caption tag with metadata"""
    tag: str = Field(..., description="Tag text")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(
        default="joycaption",
        description="Source of the tag (joycaption, aligned, custom)"
    )
    stash_tag_id: Optional[str] = Field(
        default=None,
        description="Aligned Stash tag ID if taxonomy alignment enabled"
    )
    category: Optional[str] = Field(
        default=None,
        description="Tag category (action, setting, person, object, etc.)"
    )


class FrameCaption(BaseModel):
    """Caption results for a single frame"""
    frame_index: int
    timestamp: float
    raw_caption: str = Field(..., description="Raw JoyCaption output")
    tags: List[CaptionTag] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    scene_index: Optional[int] = None
    prompt_type_used: PromptType


class SceneCaptionSummary(BaseModel):
    """Aggregated caption summary for a scene"""
    scene_index: int
    start_timestamp: float
    end_timestamp: float
    dominant_tags: List[str]
    frame_count: int
    avg_confidence: float
    combined_caption: Optional[str] = None


class CaptionMetadata(BaseModel):
    """Processing metadata"""
    source: str
    source_type: str = "video"
    total_frames: int
    frames_captioned: int
    model: str = "joycaption"
    model_variant: str = "alpha-two"
    quantization: str = "4-bit"
    prompt_type: PromptType
    processing_time_seconds: float
    device: str
    vram_peak_mb: Optional[float] = None
    gpu_wait_time_seconds: Optional[float] = None


class CaptionOutcome(BaseModel):
    """Full captioning results"""
    frames: List[FrameCaption]
    scene_summaries: Optional[List[SceneCaptionSummary]] = None
    metadata: CaptionMetadata


class CaptionResults(BaseModel):
    """Complete job results"""
    job_id: str
    source_id: str
    status: JobStatus
    captions: CaptionOutcome
    metadata: CaptionMetadata


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "captioning-service"
    version: str = "1.0.0"
    implemented: bool = True
    phase: int = 3
    message: Optional[str] = None
    model: Optional[str] = None
    model_loaded: bool = False
    device: Optional[str] = None
    vram_available_mb: Optional[float] = None
    gpu_acquired: bool = False
    default_min_confidence: float = 0.5


class TagTaxonomyNode(BaseModel):
    """A node in the Stash tag taxonomy"""
    id: str
    name: str
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    category: Optional[str] = None


class TaxonomyResponse(BaseModel):
    """Tag taxonomy for alignment"""
    tags: List[TagTaxonomyNode]
    categories: List[str]
    total_tags: int
    last_synced: str
