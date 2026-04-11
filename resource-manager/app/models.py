"""
Resource Manager Service - Pydantic Models
Request/response schemas for GPU resource orchestration
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ResourceStatus(str, Enum):
    """Resource allocation status"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    REQUESTED = "requested"
    RELEASING = "releasing"


class RequestStatus(str, Enum):
    """GPU request status"""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class GPURequestInput(BaseModel):
    """Request for GPU access"""
    service_name: str = Field(
        ...,
        description="Name of the requesting service"
    )
    vram_required_mb: float = Field(
        ...,
        ge=0,
        description="VRAM required in MB"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority (1=highest, 10=lowest)"
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Maximum wait time in seconds"
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Associated job ID for tracking"
    )
    perpetual: bool = Field(
        default=False,
        description="Perpetual lease that never expires (for always-loaded models)"
    )


class GPURequestResponse(BaseModel):
    """Response to GPU request"""
    request_id: str
    granted: bool
    lease_id: Optional[str] = None
    queue_position: Optional[int] = None
    estimated_wait_seconds: Optional[float] = None
    message: str


class GPUReleaseInput(BaseModel):
    """Release GPU access"""
    lease_id: str = Field(
        ...,
        description="Lease ID to release"
    )


class GPUReleaseResponse(BaseModel):
    """Response to GPU release"""
    released: bool
    message: str


class GPUHeartbeatInput(BaseModel):
    """Heartbeat to keep lease alive"""
    lease_id: str = Field(
        ...,
        description="Lease ID to refresh"
    )


class GPUHeartbeatResponse(BaseModel):
    """Response to heartbeat"""
    success: bool
    lease_valid: bool
    expires_in_seconds: float
    message: str


class GPULease(BaseModel):
    """Active GPU lease"""
    lease_id: str
    service_name: str
    vram_allocated_mb: float
    granted_at: str
    expires_at: str
    last_heartbeat: str
    job_id: Optional[str] = None


class GPUQueueEntry(BaseModel):
    """Entry in GPU request queue"""
    request_id: str
    service_name: str
    vram_required_mb: float
    priority: int
    requested_at: str
    timeout_at: str
    position: int


class GPUStatusResponse(BaseModel):
    """Current GPU resource status"""
    status: ResourceStatus
    total_vram_mb: float
    available_vram_mb: float
    allocated_vram_mb: float
    active_leases: List[GPULease]
    queue_length: int
    queue: List[GPUQueueEntry]


class DeviceType(str, Enum):
    """Compute device type"""
    CUDA = "cuda"       # NVIDIA GPU
    MPS = "mps"         # Apple Silicon GPU
    ROCm = "rocm"       # AMD GPU
    CPU = "cpu"         # CPU fallback
    VIRTUAL = "virtual" # Virtual/simulated for testing


class GPUInfo(BaseModel):
    """GPU/compute device hardware information"""
    device_name: str
    device_type: DeviceType
    device_index: int = 0
    total_memory_mb: float
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None

    # Backwards compatibility
    @property
    def total_vram_mb(self) -> float:
        return self.total_memory_mb


class ServiceRegistration(BaseModel):
    """Service registration for resource management"""
    service_name: str
    service_url: str
    max_vram_mb: float = Field(
        ...,
        description="Maximum VRAM this service may use"
    )
    can_be_preempted: bool = Field(
        default=True,
        description="Whether this service can be asked to release GPU"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str = "resource-manager"
    version: str = "1.0.0"
    gpu_available: bool
    gpu_info: Optional[GPUInfo] = None
    active_leases: int = 0
    queue_length: int = 0
