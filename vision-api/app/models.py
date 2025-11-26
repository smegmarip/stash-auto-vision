"""
Vision API - Data Models
Pydantic models for orchestrator API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModuleConfig(BaseModel):
    """Configuration for a single module"""
    enabled: bool = Field(default=True)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ModulesConfig(BaseModel):
    """Module configurations"""
    scenes: ModuleConfig = Field(default_factory=lambda: ModuleConfig(enabled=True))
    faces: ModuleConfig = Field(default_factory=lambda: ModuleConfig(enabled=True))
    semantics: ModuleConfig = Field(default_factory=lambda: ModuleConfig(enabled=False))
    objects: ModuleConfig = Field(default_factory=lambda: ModuleConfig(enabled=False))


class AnalyzeVideoRequest(BaseModel):
    """Request to analyze video with all services"""
    source: str = Field(..., description="Path, URL, or image source to analyze")
    scene_id: str = Field(..., description="Scene ID for reference")
    job_id: Optional[str] = Field(default=None, description="Job ID for tracking")
    processing_mode: str = Field(default="sequential", description="sequential or parallel")
    modules: ModulesConfig = Field(default_factory=ModulesConfig)


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
    stage: Optional[str] = None
    message: Optional[str] = None
    services: List[ServiceJobInfo]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AnalyzeJobResults(BaseModel):
    """Complete job results from all services.

    Service-specific results may have different structures:
    - faces: List of face detection results (from faces-service) or Dict (from vision orchestrator)
    - scenes: List of scene boundaries or Dict
    - semantics: Dict of semantic classifications
    - objects: List of detected objects or Dict
    """
    job_id: str
    scene_id: str
    status: str
    scenes: Optional[Any] = None
    faces: Optional[Any] = None
    semantics: Optional[Any] = None
    objects: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "vision-api"
    version: str = "1.0.0"
    services: Dict[str, Dict[str, Any]]


class JobSummary(BaseModel):
    """Summary of a job for listing"""
    job_id: str
    service: str = Field(..., description="Service that owns the job (vision, faces, scenes)")
    status: str
    progress: float = 0.0
    source: Optional[str] = None
    scene_id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = Field(default=None, description="Full results when include_results=true")


class ListJobsResponse(BaseModel):
    """Response for job listing endpoint"""
    jobs: List[JobSummary]
    total: int = Field(..., description="Total number of jobs matching filters (before pagination)")
    limit: int
    offset: int
