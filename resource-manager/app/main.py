"""
Resource Manager Service - Main Application
FastAPI server for GPU/VRAM resource orchestration
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import logging

from .models import (
    GPURequestInput,
    GPURequestResponse,
    GPUReleaseInput,
    GPUReleaseResponse,
    GPUHeartbeatInput,
    GPUHeartbeatResponse,
    GPUStatusResponse,
    GPUInfo,
    DeviceType,
    HealthResponse,
    ResourceStatus
)
from .gpu_manager import GPUManager

# Environment configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TOTAL_VRAM_MB = float(os.getenv("TOTAL_VRAM_MB", "16384"))  # 16GB default
LEASE_DURATION_SECONDS = float(os.getenv("LEASE_DURATION_SECONDS", "600"))
HEARTBEAT_TIMEOUT_SECONDS = float(os.getenv("HEARTBEAT_TIMEOUT_SECONDS", "60"))
# Force a specific device type (cuda, mps, cpu, virtual)
FORCE_DEVICE_TYPE = os.getenv("FORCE_DEVICE_TYPE", "").lower()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global GPU manager
gpu_manager: Optional[GPUManager] = None
gpu_info: Optional[GPUInfo] = None


def detect_compute_device() -> tuple[Optional[GPUInfo], float]:
    """
    Detect available compute device (GPU or CPU)

    Checks in order: CUDA (NVIDIA) -> MPS (Apple Silicon) -> ROCm (AMD) -> CPU

    Returns:
        Tuple of (GPUInfo or None, total_memory_mb)
    """
    # Check for forced device type
    if FORCE_DEVICE_TYPE == "virtual":
        logger.info(f"Using virtual device with {TOTAL_VRAM_MB}MB memory")
        return GPUInfo(
            device_name="Virtual Device",
            device_type=DeviceType.VIRTUAL,
            device_index=0,
            total_memory_mb=TOTAL_VRAM_MB,
        ), TOTAL_VRAM_MB

    if FORCE_DEVICE_TYPE == "cpu":
        return _get_cpu_info()

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, using CPU mode with virtual memory limit")
        return GPUInfo(
            device_name="CPU (PyTorch not available)",
            device_type=DeviceType.CPU,
            device_index=0,
            total_memory_mb=TOTAL_VRAM_MB,
        ), TOTAL_VRAM_MB

    # Try CUDA (NVIDIA)
    if FORCE_DEVICE_TYPE in ("", "cuda") and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / (1024 * 1024)
            return GPUInfo(
                device_name=props.name,
                device_type=DeviceType.CUDA,
                device_index=0,
                total_memory_mb=total_mb,
                driver_version=torch.version.cuda or "unknown",
                compute_capability=f"{props.major}.{props.minor}"
            ), total_mb
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")

    # Try MPS (Apple Silicon)
    if FORCE_DEVICE_TYPE in ("", "mps") and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # MPS doesn't expose memory info directly, estimate from system
            total_mb = _get_apple_gpu_memory()
            return GPUInfo(
                device_name="Apple Silicon GPU",
                device_type=DeviceType.MPS,
                device_index=0,
                total_memory_mb=total_mb,
                driver_version=f"macOS MPS",
            ), total_mb
        except Exception as e:
            logger.warning(f"MPS detection failed: {e}")

    # Try ROCm (AMD) - detected via CUDA-like interface in PyTorch
    if FORCE_DEVICE_TYPE in ("", "rocm"):
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip:
                if torch.cuda.is_available():  # ROCm uses CUDA interface
                    props = torch.cuda.get_device_properties(0)
                    total_mb = props.total_memory / (1024 * 1024)
                    return GPUInfo(
                        device_name=props.name,
                        device_type=DeviceType.ROCm,
                        device_index=0,
                        total_memory_mb=total_mb,
                        driver_version=torch.version.hip or "unknown",
                    ), total_mb
        except Exception as e:
            logger.warning(f"ROCm detection failed: {e}")

    # Fall back to CPU
    return _get_cpu_info()


def _get_apple_gpu_memory() -> float:
    """
    Estimate Apple Silicon GPU memory

    On unified memory Macs, GPU can use a portion of system RAM.
    Default to ~75% of system memory or configured limit.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True
        )
        total_ram_bytes = int(result.stdout.strip())
        total_ram_mb = total_ram_bytes / (1024 * 1024)
        # Apple recommends ~75% for GPU in unified memory
        gpu_memory_mb = min(total_ram_mb * 0.75, TOTAL_VRAM_MB)
        logger.info(f"Apple Silicon: {total_ram_mb:.0f}MB RAM, allocating {gpu_memory_mb:.0f}MB for GPU")
        return gpu_memory_mb
    except Exception as e:
        logger.warning(f"Could not detect Apple memory: {e}")
        return TOTAL_VRAM_MB


def _get_cpu_info() -> tuple[GPUInfo, float]:
    """Get CPU info as fallback compute device"""
    try:
        import platform
        cpu_name = platform.processor() or "Unknown CPU"
    except Exception:
        cpu_name = "CPU"

    logger.info(f"Using CPU mode with {TOTAL_VRAM_MB}MB virtual memory limit")
    return GPUInfo(
        device_name=cpu_name,
        device_type=DeviceType.CPU,
        device_index=0,
        total_memory_mb=TOTAL_VRAM_MB,
    ), TOTAL_VRAM_MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global gpu_manager, gpu_info

    logger.info("Starting Resource Manager Service...")

    # Detect compute device (GPU or CPU)
    gpu_info, total_memory = detect_compute_device()

    device_desc = f"{gpu_info.device_type.value.upper()}: {gpu_info.device_name}"
    logger.info(f"Compute device: {device_desc} ({total_memory:.0f}MB)")

    # Initialize GPU manager (works for any device type)
    gpu_manager = GPUManager(
        total_vram_mb=total_memory,
        lease_duration_seconds=LEASE_DURATION_SECONDS,
        heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS
    )
    await gpu_manager.start()

    logger.info(f"Resource Manager initialized with {total_memory:.0f}MB memory budget")

    yield

    # Cleanup
    logger.info("Shutting down Resource Manager Service...")
    if gpu_manager:
        await gpu_manager.stop()
    logger.info("Resource Manager Service stopped")


app = FastAPI(
    title="Resource Manager Service",
    description="GPU/VRAM resource orchestration for Vision services",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/resources/gpu/request", response_model=GPURequestResponse)
async def request_gpu(request: GPURequestInput):
    """
    Request GPU access

    Services call this to request exclusive GPU access.
    If GPU is available and has enough VRAM, access is granted immediately.
    Otherwise, the request is queued based on priority.
    """
    try:
        result = await gpu_manager.request_gpu(
            service_name=request.service_name,
            vram_required_mb=request.vram_required_mb,
            priority=request.priority,
            timeout_seconds=request.timeout_seconds,
            job_id=request.job_id
        )

        return GPURequestResponse(
            request_id=result["request_id"],
            granted=result["granted"],
            lease_id=result.get("lease_id"),
            queue_position=result.get("queue_position"),
            estimated_wait_seconds=result.get("estimated_wait_seconds"),
            message=result["message"]
        )

    except Exception as e:
        logger.error(f"Error processing GPU request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/gpu/request/{request_id}")
async def get_request_status(request_id: str):
    """
    Get status of a GPU request

    Poll this endpoint to check if a queued request has been granted.
    """
    try:
        result = await gpu_manager.get_request_status(request_id)
        return result

    except Exception as e:
        logger.error(f"Error getting request status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resources/gpu/wait/{request_id}")
async def wait_for_gpu(request_id: str):
    """
    Wait for GPU access to be granted

    Blocks until the request is granted or times out.
    """
    try:
        result = await gpu_manager.wait_for_grant(request_id)

        if not result["granted"]:
            raise HTTPException(status_code=408, detail=result["message"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error waiting for GPU: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resources/gpu/release", response_model=GPUReleaseResponse)
async def release_gpu(request: GPUReleaseInput):
    """
    Release GPU access

    Services must call this when done with GPU to allow other services to use it.
    """
    try:
        result = await gpu_manager.release_gpu(request.lease_id)

        if not result["released"]:
            raise HTTPException(status_code=404, detail=result["message"])

        return GPUReleaseResponse(
            released=result["released"],
            message=result["message"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing GPU: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resources/gpu/heartbeat", response_model=GPUHeartbeatResponse)
async def heartbeat(request: GPUHeartbeatInput):
    """
    Send heartbeat to keep lease alive

    Services must send periodic heartbeats to prevent lease expiration.
    """
    try:
        result = await gpu_manager.heartbeat(request.lease_id)

        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])

        return GPUHeartbeatResponse(
            success=result["success"],
            lease_valid=result["lease_valid"],
            expires_in_seconds=result["expires_in_seconds"],
            message=result["message"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing heartbeat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/gpu/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """
    Get current GPU resource status

    Returns information about active leases, available VRAM, and queue.
    """
    try:
        status = gpu_manager.get_status()

        return GPUStatusResponse(
            status=ResourceStatus(status["status"]),
            total_vram_mb=status["total_vram_mb"],
            available_vram_mb=status["available_vram_mb"],
            allocated_vram_mb=status["allocated_vram_mb"],
            active_leases=status["active_leases"],
            queue_length=status["queue_length"],
            queue=status["queue"]
        )

    except Exception as e:
        logger.error(f"Error getting GPU status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resources/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = gpu_manager.get_status() if gpu_manager else {}

    # GPU is "available" if we have a real GPU (not CPU/virtual)
    has_gpu = gpu_info is not None and gpu_info.device_type not in (DeviceType.CPU, DeviceType.VIRTUAL)

    return HealthResponse(
        status="healthy",
        gpu_available=has_gpu,
        gpu_info=gpu_info,
        active_leases=len(status.get("active_leases", [])),
        queue_length=status.get("queue_length", 0)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5007)
