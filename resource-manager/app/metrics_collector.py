"""
Resource Manager - System Metrics Collector

Samples GPU utilization, VRAM usage, CPU usage, and RAM usage at a
configurable interval and maintains a rolling 1-hour history buffer.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum history: 1 hour at 5-second intervals = 720 samples
MAX_HISTORY = 720


@dataclass
class MetricSample:
    """Single point-in-time metric sample."""
    timestamp: float
    gpu_utilization_pct: float  # 0-100
    vram_used_mb: float
    vram_total_mb: float
    cpu_utilization_pct: float  # 0-100
    ram_used_mb: float
    ram_total_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 1),
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_total_mb": round(self.ram_total_mb, 1),
        }


class MetricsCollector:
    """Collects and buffers system metrics for the /resources/metrics endpoint."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._history: deque[MetricSample] = deque(maxlen=MAX_HISTORY)
        self._task: Optional[asyncio.Task] = None
        self._nvml_initialized = False
        self._psutil_available = False
        self._gpu_handle = None

        self._init_backends()

    def _init_backends(self) -> None:
        """Try to initialize pynvml and psutil."""
        # GPU metrics via pynvml (bundled with nvidia-ml-py3 / torch)
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialized = True
            logger.info("Metrics: pynvml initialized for GPU metrics")
        except Exception as e:
            logger.warning(f"Metrics: pynvml unavailable ({e}), GPU metrics will be zeros")

        # System metrics via psutil
        try:
            import psutil  # noqa: F401
            self._psutil_available = True
            # Prime the first cpu_percent call (returns 0.0 on first call)
            psutil.cpu_percent(interval=None)
            logger.info("Metrics: psutil available for CPU/RAM metrics")
        except ImportError:
            logger.warning("Metrics: psutil not installed, CPU/RAM metrics will be zeros")

    def _sample(self) -> MetricSample:
        """Collect a single metric sample from all backends."""
        gpu_util = 0.0
        vram_used = 0.0
        vram_total = 0.0
        cpu_util = 0.0
        ram_used = 0.0
        ram_total = 0.0

        if self._nvml_initialized:
            try:
                import pynvml
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = float(rates.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_used = mem_info.used / (1024 * 1024)
                vram_total = mem_info.total / (1024 * 1024)
            except Exception as e:
                logger.debug(f"GPU sample error: {e}")

        if self._psutil_available:
            try:
                import psutil
                cpu_util = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                ram_used = mem.used / (1024 * 1024)
                ram_total = mem.total / (1024 * 1024)
            except Exception as e:
                logger.debug(f"System sample error: {e}")

        return MetricSample(
            timestamp=time.time(),
            gpu_utilization_pct=gpu_util,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            cpu_utilization_pct=cpu_util,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
        )

    def get_actual_vram(self) -> Tuple[float, float]:
        """Return (used_mb, total_mb) from hardware. For use by GPUManager."""
        if not self._nvml_initialized:
            return 0.0, 0.0
        try:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            return mem_info.used / (1024 * 1024), mem_info.total / (1024 * 1024)
        except Exception:
            return 0.0, 0.0

    async def start(self) -> None:
        """Start the background sampling loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            logger.info(f"Metrics collector started (interval={self.interval}s, max_history={MAX_HISTORY})")

    async def stop(self) -> None:
        """Stop the background sampling loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("Metrics collector stopped")

    async def _loop(self) -> None:
        """Sample metrics at the configured interval."""
        while True:
            try:
                sample = self._sample()
                self._history.append(sample)
            except Exception as e:
                logger.error(f"Metrics sample error: {e}")
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    def get_metrics(self) -> Dict[str, Any]:
        """Return current values, averages, and time-series history."""
        history = list(self._history)

        if not history:
            return {
                "current": None,
                "averages": None,
                "history": [],
                "sample_count": 0,
                "interval_seconds": self.interval,
            }

        current = history[-1]

        # Compute averages over the buffer
        n = len(history)
        avg_gpu = sum(s.gpu_utilization_pct for s in history) / n
        avg_vram = sum(s.vram_used_mb for s in history) / n
        avg_cpu = sum(s.cpu_utilization_pct for s in history) / n
        avg_ram = sum(s.ram_used_mb for s in history) / n

        return {
            "current": current.to_dict(),
            "averages": {
                "gpu_utilization_pct": round(avg_gpu, 1),
                "vram_used_mb": round(avg_vram, 1),
                "vram_total_mb": current.vram_total_mb,
                "cpu_utilization_pct": round(avg_cpu, 1),
                "ram_used_mb": round(avg_ram, 1),
                "ram_total_mb": current.ram_total_mb,
            },
            "history": [s.to_dict() for s in history],
            "sample_count": n,
            "interval_seconds": self.interval,
        }
