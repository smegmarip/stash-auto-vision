"""
GPU Subprocess Worker

Runs the entire GPU pipeline (classifier + JoyCaption + Llama) in a
single isolated subprocess sharing one CUDA context.  This matches the
old in-process ModelManager layout — bitsandbytes can reuse its leaked
allocations within the process — but with a clean exit to reclaim all
VRAM when the job finishes or the lease is evicted.
"""

import asyncio
import logging
import multiprocessing as mp
import traceback
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _gpu_worker(input_queue: mp.Queue, result_queue: mp.Queue, progress_queue: mp.Queue):
    """Subprocess entry point. Loads all models in one CUDA context, runs full pipeline, exits."""
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
        device: str = task.get("device", "cuda")
        cache_dir: Optional[str] = task.get("cache_dir")

        # Classifier config
        classifier_model_variant: str = task.get("classifier_model_variant", "text-only")
        classifier_device: str = task.get("classifier_device", device)
        taxonomy: dict = task.get("taxonomy", {})
        classifier_params: dict = task.get("classifier_params", {})
        generate_embeddings: bool = task.get("generate_embeddings", False)

        # Load frame images
        progress_queue.put(("progress", 0.16, "Loading frames"))
        frame_images = []
        for p in frame_paths:
            try:
                frame_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                logger.warning(f"Failed to load frame {p}: {e}")
                frame_images.append(Image.new("RGB", (512, 512)))

        # --- Step 1: JoyCaption ---
        progress_queue.put(("progress", 0.18, "Loading JoyCaption model"))
        from .caption_generator import CaptionGenerator

        caption_gen = CaptionGenerator(use_quantization=True, device=device, cache_dir=cache_dir)
        caption_gen.load()

        progress_queue.put(("progress", 0.20, "Generating frame captions"))
        raw_captions = caption_gen.generate_captions(frame_images)
        fixed_captions = [CaptionGenerator.fix_caption(c, i) for i, c in enumerate(raw_captions)]

        caption_gen.unload()
        del caption_gen

        progress_queue.put(("progress", 0.55, f"Generated {len(fixed_captions)} captions"))

        # --- Step 2: Llama summary + title ---
        progress_queue.put(("progress", 0.58, "Loading Llama model"))
        from .llama_runtime import LlamaRuntime
        from .summary_generator import SummaryGenerator
        from .title_generator import TitleGenerator

        llm = LlamaRuntime(device=device, cache_dir=cache_dir)
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
                frame_caption_dicts, promo_desc, duration,
                performer_count, performer_genders, resolution,
            )
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            scene_summary = " ".join(fixed_captions)

        progress_queue.put(("progress", 0.80, "Generating scene title"))
        suggested_title = None
        try:
            suggested_title = title_gen.generate_title(
                scene_source=source, scene_summary=scene_summary,
                promo_desc=promo_desc, duration=duration,
                performer_count=performer_count,
                performer_genders=performer_genders,
                resolution=resolution,
            )
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")

        llm.unload()
        del llm, summary_gen, title_gen

        # --- Step 3: Load classifier and run (after JoyCaption + Llama are done) ---
        progress_queue.put(("progress", 0.85, "Loading tag classifier"))
        from .classifier import TagClassifier

        tag_classifier = TagClassifier(model_variant=classifier_model_variant, device=classifier_device)
        tag_classifier.load_model()
        if taxonomy and taxonomy.get("tags"):
            tag_classifier.load_taxonomy(taxonomy)
        logger.info(f"Classifier loaded in subprocess: {classifier_model_variant}")

        progress_queue.put(("progress", 0.88, "Running tag classifier"))
        prediction = tag_classifier.predict(
            frame_captions=fixed_captions,
            summary=scene_summary,
            promo_desc=classifier_params.get("promo_desc", promo_desc),
            has_promo=classifier_params.get("has_promo", bool(promo_desc)),
            top_k=classifier_params.get("top_k", 50),
            min_score=classifier_params.get("min_score", 0.75),
            use_hierarchical_decoding=classifier_params.get("use_hierarchical_decoding", True),
        )
        tags = prediction["tags"]

        scene_embedding = None
        if generate_embeddings:
            try:
                scene_embedding = tag_classifier.get_scene_embedding(
                    fixed_captions, scene_summary,
                    classifier_params.get("promo_desc", promo_desc),
                    classifier_params.get("has_promo", bool(promo_desc)),
                )
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        tag_classifier.unload()
        del tag_classifier

        progress_queue.put(("progress", 0.95, f"Classified {len(tags)} tags"))

        result_queue.put(("success", {
            "captions": fixed_captions,
            "summary": scene_summary,
            "title": suggested_title,
            "tags": tags,
            "scene_embedding": scene_embedding,
        }))

    except Exception as e:
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


async def run_gpu_pipeline(
    frame_paths: List[str],
    frame_timestamps: List[float],
    promo_desc: str = "",
    duration: float = 0,
    performer_count: int = 0,
    performer_genders: Optional[List[str]] = None,
    resolution: str = "Unknown",
    source: str = "",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    classifier_model_variant: str = "text-only",
    classifier_device: str = "cuda",
    taxonomy: Optional[dict] = None,
    classifier_params: Optional[dict] = None,
    generate_embeddings: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Run the full GPU pipeline in an isolated subprocess.

    Returns dict with keys: captions, summary, title, tags, scene_embedding.
    Raises RuntimeError if the subprocess fails.
    """
    ctx = mp.get_context("spawn")
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
        "device": device,
        "cache_dir": cache_dir,
        "classifier_model_variant": classifier_model_variant,
        "classifier_device": classifier_device,
        "taxonomy": taxonomy or {},
        "classifier_params": classifier_params or {},
        "generate_embeddings": generate_embeddings,
    })

    proc = ctx.Process(target=_gpu_worker, args=(input_queue, result_queue, progress_queue))
    proc.start()
    logger.info(f"GPU subprocess started (PID {proc.pid})")

    while proc.is_alive() or not result_queue.empty():
        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                if msg[0] == "progress" and progress_callback:
                    try:
                        await progress_callback(msg[1], msg[2])
                    except Exception:
                        pass
            except Exception:
                pass
        if not result_queue.empty():
            break
        await asyncio.sleep(1.0)

    while not progress_queue.empty():
        try:
            msg = progress_queue.get_nowait()
            if msg[0] == "progress" and progress_callback:
                try:
                    await progress_callback(msg[1], msg[2])
                except Exception:
                    pass
        except Exception:
            pass

    proc.join(timeout=900)
    if proc.is_alive():
        logger.error("GPU subprocess did not exit after 15 min, terminating")
        proc.terminate()
        proc.join(timeout=10)

    logger.info(f"GPU subprocess exited (code={proc.exitcode})")

    if result_queue.empty():
        raise RuntimeError(f"GPU subprocess exited without results (code={proc.exitcode})")

    status, data = result_queue.get()
    if status == "error":
        raise RuntimeError(f"GPU subprocess failed: {data}")

    return data
