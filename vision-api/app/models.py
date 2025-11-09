"""
Vision API - Data Models
Pydantic models for orchestrator API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AnalyzeVideoRequest(BaseModel):
    """Request to analyze video with all services"""
    video_path: str = Field(..., description="Absolute path to video file")
    scene_id: str = Field(..., description="Scene ID for reference")
    job_id: Optional[str] = Field(default=None, description="Job ID for tracking")
    enable_scenes: bool = Field(default=True)
    enable_faces: bool = Field(default=True)
    enable_semantics: bool = Field(default=False, description="Phase 2 - not implemented")
    enable_objects: bool = Field(default=False, description="Phase 3 - not implemented")
    processing_mode: str = Field(default="sequential", description="sequential or parallel")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeJobResponse(BaseModel):
    """Response for video analysis job submission"""
    job_id: str
    status: str
    created_at: str
    services_enabled: Dict[str, bool]
    processing_mode: str


class ServiceJobInfo(BaseModel):
    """Information about a service job"""
    service: str
    job_id: Optional[str] = None
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None


class AnalyzeJobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float
    processing_mode: str
    services: List[ServiceJobInfo]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class AnalyzeJobResults(BaseModel):
    """Complete job results from all services"""
    job_id: str
    scene_id: str
    status: str
    scenes: Optional[Dict[str, Any]] = None
    faces: Optional[Dict[str, Any]] = None
    semantics: Optional[Dict[str, Any]] = None
    objects: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "vision-api"
    version: str = "1.0.0"
    services: Dict[str, Dict[str, Any]]
