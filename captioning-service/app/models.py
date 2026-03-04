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
    SCENE_SUMMARY = "scene_summary"  # Detailed JSON scene analysis


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
    use_hierarchical_scoring: bool = Field(
        default=True,
        description="Use DFS hierarchical scoring for tag matching"
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
    select_sharpest: bool = Field(
        default=True,
        description="Select sharpest frames per scene using Laplacian variance"
    )
    sharpness_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Extract N*frames_per_scene candidates for sharpness selection"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt override (advanced)"
    )
    sprite_vtt_url: Optional[str] = Field(
        default=None,
        description="URL to sprite VTT file (for sprite_sheet frame selection)"
    )
    sprite_image_url: Optional[str] = Field(
        default=None,
        description="URL to sprite grid image (for sprite_sheet frame selection)"
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


class PersonDetail(BaseModel):
    """Details about a person in the scene"""
    gender: Optional[str] = None
    age_range: Optional[str] = None  # child, teen, young_adult, adult, middle_aged, elderly
    ethnicity: Optional[str] = None
    body_type: Optional[str] = None
    hair: Optional[str] = None  # color, length, style
    expression: Optional[str] = None  # emotional expression
    pose: Optional[str] = None  # standing, sitting, lying, etc.
    position: Optional[str] = None  # foreground, background, left, right, center
    description: Optional[str] = None


class CinematographyInfo(BaseModel):
    """Cinematography details"""
    shot_type: Optional[str] = None  # extreme_close_up, close_up, medium_close_up, medium, medium_wide, wide, extreme_wide
    camera_angle: Optional[str] = None  # eye_level, high_angle, low_angle, dutch_angle, birds_eye, worms_eye, over_shoulder, pov
    camera_movement: Optional[str] = None  # static, pan, tilt, tracking, dolly, crane, handheld, steadicam, zoom
    focus: Optional[str] = None  # shallow_dof, deep_dof, rack_focus, soft_focus, sharp
    composition: Optional[str] = None  # rule_of_thirds, centered, symmetrical, asymmetrical, diagonal, leading_lines
    framing: Optional[str] = None  # full_body, three_quarter, half_body, head_and_shoulders, face_only


class VisualStyle(BaseModel):
    """Visual style and quality information"""
    color_palette: List[str] = Field(default_factory=list)  # dominant colors
    color_grading: Optional[str] = None  # warm, cool, neutral, desaturated, vibrant, monochrome
    contrast: Optional[str] = None  # high, low, normal
    saturation: Optional[str] = None  # saturated, desaturated, muted, vivid
    film_grain: Optional[str] = None  # none, light, heavy, digital_noise
    quality: Optional[str] = None  # hd, sd, 4k, vintage, degraded
    visual_style: Optional[str] = None  # cinematic, documentary, amateur, professional, artistic
    era_aesthetic: Optional[str] = None  # modern, 90s, 80s, 70s, vintage, retro


class EnvironmentInfo(BaseModel):
    """Environment and atmosphere details"""
    time_of_day: Optional[str] = None  # morning, afternoon, evening, night, golden_hour, blue_hour
    weather: Optional[str] = None  # sunny, cloudy, rainy, snowy, foggy, stormy
    season: Optional[str] = None  # spring, summer, fall, winter
    atmosphere: Optional[str] = None  # tense, relaxed, romantic, mysterious, energetic, somber
    ambient_light: Optional[str] = None  # bright, dim, dark, mixed


class SceneSummaryData(BaseModel):
    """Structured scene summary from VLM analysis"""
    # Location
    locale: Optional[str] = None  # indoor/outdoor + geographic type
    setting: Optional[str] = None  # specific environment (bedroom, office, beach)
    location_details: Optional[str] = None  # additional location specifics

    # People
    persons: Optional[Dict[str, Any]] = None  # {"count": int, "details": [PersonDetail]}
    attire: List[str] = Field(default_factory=list)  # clothing items
    interactions: Optional[str] = None  # how people are interacting

    # Objects and scene elements
    objects: List[str] = Field(default_factory=list)  # notable props/items
    furniture: List[str] = Field(default_factory=list)  # furniture visible
    background_elements: List[str] = Field(default_factory=list)
    foreground_elements: List[str] = Field(default_factory=list)
    text_visible: Optional[str] = None  # any visible text/signage

    # Actions
    activities: List[str] = Field(default_factory=list)  # actions being performed
    action_intensity: Optional[str] = None  # static, mild, moderate, intense

    # Technical
    cinematography: Optional[CinematographyInfo] = None
    visual_style: Optional[VisualStyle] = None
    environment: Optional[EnvironmentInfo] = None
    lighting: Optional[str] = None  # detailed lighting description
    lighting_type: Optional[str] = None  # natural, artificial, mixed, practical, studio

    # Mood and genre
    mood: Optional[str] = None  # emotional tone
    tension_level: Optional[str] = None  # none, low, medium, high
    genre: Optional[str] = None  # primary film genre
    sub_genre: Optional[str] = None  # specific genre classification
    content_type: Optional[str] = None  # narrative, documentary, music_video, interview, etc.

    # Additional context
    narrative_context: Optional[str] = None  # what seems to be happening story-wise
    notable_features: List[str] = Field(default_factory=list)  # anything unusual or distinctive


class FrameCaption(BaseModel):
    """Caption results for a single frame"""
    frame_index: int
    timestamp: float
    raw_caption: str = Field(..., description="Raw JoyCaption output")
    tags: List[CaptionTag] = Field(default_factory=list)
    summary: Optional[SceneSummaryData] = Field(
        default=None,
        description="Structured scene summary (when using SCENE_SUMMARY prompt)"
    )
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
    description: Optional[str] = Field(
        default=None,
        description="Disambiguating description (e.g., 'tails' -> 'animal appendage, not coin flip')"
    )


class TaxonomyResponse(BaseModel):
    """Tag taxonomy for alignment"""
    tags: List[TagTaxonomyNode]
    categories: List[str]
    total_tags: int
    last_synced: str
