"""
GPU model lifecycle manager with idle-timeout unloading.

Keeps models loaded across requests and unloads them after a configurable
idle period. This avoids the ~2-minute load time on consecutive requests
while still freeing VRAM when the service is idle.

Models are loaded on first use and stay resident until idle_timeout_seconds
elapses with no activity. A background task checks periodically.

The classifier is always kept loaded (lightweight, ~1.4GB). JoyCaption
(~8GB) and the summary model (~7GB) cannot coexist in VRAM on a single
16GB GPU, so they are loaded/unloaded as needed — but kept resident
between consecutive jobs.
"""

import asyncio
import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages GPU model lifecycles with idle-timeout unloading.

    Usage:
        manager = ModelManager(idle_timeout=300)
        manager.register("caption", caption_gen)
        manager.register("summary", summary_gen)

        # In request handler:
        manager.ensure_loaded("caption")  # loads if needed, resets idle timer
        caption_gen.generate_captions(...)
        manager.mark_done("caption")      # starts idle timer

        # When switching to a model that conflicts in VRAM:
        manager.ensure_loaded("summary")  # auto-unloads "caption" if needed
    """

    def __init__(
        self,
        idle_timeout: float = 300.0,
        check_interval: float = 30.0,
        vram_budget_mb: float = 16000.0,
    ):
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.vram_budget_mb = vram_budget_mb
        self._models: dict[str, "_ModelEntry"] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        # Reentrant lock guards model load/unload across threads
        self._lock = threading.RLock()

    def register(self, name: str, model, vram_mb: float = 0, exclusive: bool = False):
        """Register a model for lifecycle management.

        Args:
            name: Unique model name (e.g., "caption", "summary").
            model: Object with load()/unload()/is_loaded properties.
            vram_mb: Estimated VRAM usage in MB.
            exclusive: If True, other exclusive models are unloaded before loading.
        """
        self._models[name] = _ModelEntry(
            name=name, model=model, vram_mb=vram_mb, exclusive=exclusive,
        )
        logger.info(f"Registered model '{name}' (vram={vram_mb:.0f}MB, exclusive={exclusive})")

    def ensure_loaded(self, name: str) -> None:
        """Ensure a model is loaded, unloading conflicting models if needed.

        Thread-safe — concurrent calls block until the first one completes.
        Holds the lock for the entire load duration to prevent racing loads.
        """
        with self._lock:
            entry = self._models.get(name)
            if not entry:
                raise ValueError(f"Unknown model: {name}")

            if entry.model.is_loaded:
                entry.last_used = time.monotonic()
                return

            # Unload conflicting exclusive models if this one is exclusive.
            # Refuse to unload a busy model — the caller must wait and retry.
            if entry.exclusive:
                for other_name, other in self._models.items():
                    if other_name != name and other.exclusive and other.model.is_loaded:
                        if other.busy_count > 0:
                            raise RuntimeError(
                                f"Cannot load '{name}': conflicting model '{other_name}' is busy"
                            )
                        logger.info(f"Unloading '{other_name}' to make room for '{name}'")
                        other.model.unload()
                        other.loaded_at = None

            entry.model.load()
            entry.loaded_at = time.monotonic()
            entry.last_used = time.monotonic()

    def mark_done(self, name: str) -> None:
        """Mark a model as done with current work. Starts the idle timer."""
        with self._lock:
            entry = self._models.get(name)
            if entry:
                entry.last_used = time.monotonic()

    @contextmanager
    def using(self, name: str) -> Iterator[None]:
        """Context manager: ensure model is loaded and mark it busy for the duration.

        The cleanup loop will NOT unload a busy model, preventing mid-inference
        unloads. Nested or concurrent calls increment a counter; the model is
        only eligible for idle unload when busy_count returns to zero.
        """
        self.ensure_loaded(name)
        with self._lock:
            entry = self._models[name]
            entry.busy_count += 1
        try:
            yield
        finally:
            with self._lock:
                entry = self._models.get(name)
                if entry:
                    entry.busy_count = max(0, entry.busy_count - 1)
                    entry.last_used = time.monotonic()

    def unload(self, name: str) -> None:
        """Explicitly unload a model."""
        with self._lock:
            entry = self._models.get(name)
            if entry and entry.model.is_loaded:
                entry.model.unload()
                entry.loaded_at = None

    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            for entry in self._models.values():
                if entry.model.is_loaded:
                    entry.model.unload()
                    entry.loaded_at = None

    def start_cleanup_loop(self) -> None:
        """Start the background idle-check loop."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"Model cleanup loop started (timeout={self.idle_timeout}s, interval={self.check_interval}s)")

    def stop_cleanup_loop(self) -> None:
        """Stop the background idle-check loop."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically check for idle models and unload them.

        Busy models (busy_count > 0) are skipped — unloading while inference
        is in progress would null out the tokenizer/model attributes and
        crash the running call.
        """
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                with self._lock:
                    now = time.monotonic()
                    for name, entry in self._models.items():
                        if not entry.model.is_loaded or entry.last_used is None:
                            continue
                        if entry.busy_count > 0:
                            # In use — bump last_used so it doesn't expire mid-inference
                            entry.last_used = now
                            continue
                        idle_time = now - entry.last_used
                        if idle_time > self.idle_timeout:
                            logger.info(f"Unloading idle model '{name}' (idle {idle_time:.0f}s > {self.idle_timeout}s)")
                            entry.model.unload()
                            entry.loaded_at = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in model cleanup loop")


class _ModelEntry:
    """Internal tracking for a registered model."""

    __slots__ = ("name", "model", "vram_mb", "exclusive", "loaded_at", "last_used", "busy_count")

    def __init__(self, name: str, model, vram_mb: float, exclusive: bool):
        self.name = name
        self.model = model
        self.vram_mb = vram_mb
        self.exclusive = exclusive
        self.loaded_at: Optional[float] = None
        self.last_used: Optional[float] = None
        # Number of callers currently using this model. The cleanup loop
        # will not unload a model while busy_count > 0.
        self.busy_count: int = 0
