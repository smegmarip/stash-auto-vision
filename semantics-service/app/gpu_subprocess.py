"""
GPU Subprocess Worker

Runs JoyCaption + Llama inference in an isolated subprocess so the CUDA
context is fully destroyed on exit, reclaiming all VRAM (including
bitsandbytes allocations that leak in the parent process).

The parent calls run_gpu_pipeline() which spawns a child process, passes
frame paths + metadata through a Queue, and collects results when the
child exits.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def _worker(
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """Subprocess entry point. Loads models, runs inference, puts results, exits."""
    try:
        task = input_queue.get()
        frame_paths: List[str] = task["frame_paths"]
        frame_timestamps: List[float] = task["frame_timestamps"]
        promo_desc: str = task.get("promo_desc", "")
        duration: float = task.get("duration", 0)
        performer_count: int = task.get("performer_count", 0)
        performer_genders: Optional[List[str]] = task.get("performer_genders")
        resolution: str = task.get("resolution", "Unknown")
        source: str = task.get("source", "")
        caption_model: str = task.get("caption_model", "")
        llm_model: str = task.get("llm_model", "")
        llm_device: str = task.get("llm_device", "cuda")
        cache_dir: Optional[str] = task.get("cache_dir")

        # Load frame images
        progress_queue.put(("progress", 0.16, "Loading frames"))
        frame_images = []
        for p in frame_paths:
            try:
                frame_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load frame {p}: {e}")
                frame_images.append(Image.new("RGB", (512, 512)))

        # --- JoyCaption ---
        progress_queue.put(("progress", 0.18, "Loading JoyCaption model"))
        from .caption_generator import CaptionGenerator

        caption_gen = CaptionGenerator(
            model_name=caption_model or None,
            device=llm_device,
            cache_dir=cache_dir,
        )
        caption_gen.load()

        progress_queue.put(("progress", 0.20, "Generating frame captions"))
        raw_captions = caption_gen.generate_captions(frame_images)
        fixed_captions = [CaptionGenerator.fix_caption(c, i) for i, c in enumerate(raw_captions)]

        # Free JoyCaption VRAM before loading Llama
        caption_gen.unload()
        del caption_gen

        progress_queue.put(("progress", 0.55, f"Generated {len(fixed_captions)} captions"))

        # --- Llama summary ---
        progress_queue.put(("progress", 0.58, "Loading Llama model"))
        from .llama_runtime import LlamaRuntime
        from .summary_generator import SummaryGenerator
        from .title_generator import TitleGenerator

        llm = LlamaRuntime(
            model_name=llm_model or None,
            device=llm_device,
            cache_dir=cache_dir,
        )
        llm.load()

        summary_gen = SummaryGenerator(llm)
        title_gen = TitleGenerator(llm)

        frame_caption_dicts = [
            {"frame_index": i, "timestamp": frame_timestamps[i], "caption": c}
            for i, c in enumerate(fixed_captions)
        ]

        progress_queue.put(("progress", 0.62, "Generating scene summary"))
        try:
            scene_summary = summary_gen.generate_summary(
                frame_caption_dicts,
                promo_desc,
                duration,
                performer_count,
                performer_genders,
                resolution,
            )
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            scene_summary = " ".join(fixed_captions)

        progress_queue.put(("progress", 0.80, "Generating scene title"))
        suggested_title = None
        try:
            suggested_title = title_gen.generate_title(
                scene_source=source,
                scene_summary=scene_summary,
                promo_desc=promo_desc,
                duration=duration,
                performer_count=performer_count,
                performer_genders=performer_genders,
                resolution=resolution,
            )
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")

        # Cleanup before exit (not strictly necessary since process dies,
        # but makes logs cleaner)
        llm.unload()
        del llm, summary_gen, title_gen

        result_queue.put(("success", {
            "captions": fixed_captions,
            "summary": scene_summary,
            "title": suggested_title,
        }))

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put(("error", f"{e}\n{tb}"))


async def run_gpu_pipeline(
    frame_paths: List[str],
    frame_timestamps: List[float],
    promo_desc: str = "",
    duration: float = 0,
    performer_count: int = 0,
    performer_genders: Optional[List[str]] = None,
    resolution: str = "Unknown",
    source: str = "",
    caption_model: str = "",
    llm_model: str = "",
    llm_device: str = "cuda",
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Run GPU inference in an isolated subprocess.

    Spawns a child process that loads JoyCaption + Llama, runs inference,
    returns results, and exits — fully destroying its CUDA context and
    reclaiming all VRAM.

    Args:
        frame_paths: List of file paths to frame images.
        frame_timestamps: Corresponding timestamps in seconds.
        progress_callback: Optional async-friendly callback(progress, message).

    Returns:
        Dict with keys: captions, summary, title.

    Raises:
        RuntimeError: If the subprocess fails.
    """
    ctx = mp.get_context("spawn")  # spawn ensures clean CUDA context
    input_queue = ctx.Queue()
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()

    input_queue.put({
        "frame_paths": frame_paths,
        "frame_timestamps": frame_timestamps,
        "promo_desc": promo_desc,
        "duration": duration,
        "performer_count": performer_count,
        "performer_genders": performer_genders,
        "resolution": resolution,
        "source": source,
        "caption_model": caption_model,
        "llm_model": llm_model,
        "llm_device": llm_device,
        "cache_dir": cache_dir,
    })

    proc = ctx.Process(target=_worker, args=(input_queue, result_queue, progress_queue))
    proc.start()

    logger.info(f"GPU subprocess started (PID {proc.pid})")

    # Poll for progress and results
    loop = asyncio.get_running_loop()
    while proc.is_alive() or not result_queue.empty():
        # Drain progress updates
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == "progress" and progress_callback:
                    await progress_callback(msg[1], msg[2])
            except Exception:
                pass

        # Check for results
        if not result_queue.empty():
            break

        await asyncio.sleep(0.5)

    # Final drain of progress queue
    while not progress_queue.empty():
        try:
            msg = progress_queue.get_nowait()
            if msg[0] == "progress" and progress_callback:
                await progress_callback(msg[1], msg[2])
        except Exception:
            pass

    proc.join(timeout=30)
    if proc.is_alive():
        logger.error("GPU subprocess did not exit, terminating")
        proc.terminate()
        proc.join(timeout=5)

    logger.info(f"GPU subprocess exited (code={proc.exitcode})")

    if result_queue.empty():
        raise RuntimeError(f"GPU subprocess exited without results (code={proc.exitcode})")

    status, data = result_queue.get()
    if status == "error":
        raise RuntimeError(f"GPU subprocess failed: {data}")

    return data
