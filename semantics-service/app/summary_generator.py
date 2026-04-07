"""
Scene summary generator using local Llama 3.1 8B Instruct.

Synthesizes frame-level captions and scene metadata into coherent narrative
summaries. Uses the same prompt structure as the training pipeline's
generate_scene_summaries.py to ensure consistency.

Loads/unloads per job to share GPU memory with JoyCaption and the classifier.
"""

import gc
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "RedHatAI/Llama-3.1-8B-Instruct"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are a scene summarizer. Given structured data about a video scene (metadata, participants, and frame-by-frame descriptions), generate a coherent narrative summary.

Guidelines:
- Synthesize frame descriptions into a flowing narrative, not a list
- Describe temporal progression: what happens first, then, throughout, finally
- Focus on actions, interactions, settings, and visual elements
- Ignore mentions of: watermarks, compression artifacts, image quality, "photograph" framing
- Use present tense for describing the scene
- Be specific about: clothing, positions, settings, camera angles, lighting
- Do not speculate about mood, emotions, or intent unless clearly evident
- Length: 2-4 paragraphs covering the entire scene"""

SUMMARY_PROMPT_TEMPLATE = """Summarize this video scene based on the structured data below.

## Scene Metadata
- Duration: {duration_str}
- Frame count: {frame_count} frames sampled
- Resolution: {resolution}

## Participants
{participants_str}

## Promotional Description
{promotional_summary}

## Frame-by-Frame Descriptions
{frame_descriptions}

---

Generate a coherent narrative summary of this scene:"""

# Patterns for cleaning frame captions before summarization
_CAPTION_MEDIUM_RE = re.compile(
    r"^(A|This|The) (photograph|image|picture|photo)\b",
    re.IGNORECASE,
)
_WATERMARK_RE = re.compile(r"[^.]*watermark[^.]*\.", re.IGNORECASE)
_ARTIFACT_RE = re.compile(r"[^.]*compression artifacts?[^.]*\.", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"[^.]*resolution[^.]*\.", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_frame_caption(caption: str) -> str:
    """Clean a frame caption for scene summary context."""
    caption = _CAPTION_MEDIUM_RE.sub("The frame", caption)
    caption = _WATERMARK_RE.sub("", caption)
    caption = _ARTIFACT_RE.sub("", caption)
    caption = _RESOLUTION_RE.sub("", caption)
    caption = _WHITESPACE_RE.sub(" ", caption).strip()
    return caption


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds as a human-readable string."""
    if not seconds:
        return "Unknown"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_participants(count: int, genders: Optional[List[str]]) -> str:
    """Format participant info as a descriptive string."""
    if count == 0:
        return "No identified participants"
    gender_str = ", ".join(genders) if genders else "genders unknown"
    return f"{count} participant(s): {gender_str}"


class SummaryGenerator:
    """Generates narrative scene summaries using a local Llama model.

    Follows the same load/unload pattern as CaptionGenerator to share
    GPU memory. Call load() before generating, unload() after.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("SEMANTICS_LLM_MODEL", DEFAULT_MODEL)
        self.device = device or os.getenv("SEMANTICS_LLM_DEVICE", "cuda")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._vram_mb: Optional[float] = None

    def load(self) -> None:
        """Load model and tokenizer into memory.

        On CUDA: uses 4-bit NF4 quantization to fit alongside the classifier
        in 16GB VRAM (full bfloat16 would need ~16GB on its own).
        On CPU: uses bfloat16 (native Llama precision).
        """
        if self._loaded:
            return

        logger.info("Loading summary model: %s (device=%s)", self.model_name, self.device)
        start = time.time()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if self.cache_dir:
                model_kwargs["cache_dir"] = self.cache_dir

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **model_kwargs)

            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                )
            else:
                logger.info("Using 8-bit quantization for summary model")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                )

            self.model.eval()
            self._loaded = True

            if self.device == "cuda":
                torch.cuda.synchronize()
                self._vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            elapsed = time.time() - start
            logger.info("Summary model loaded in %.1fs (device=%s, VRAM: %.0fMB)", elapsed, self.device, self._vram_mb or 0)

        except Exception:
            logger.exception("Failed to load summary model")
            raise

    def unload(self) -> None:
        """Free GPU memory by unloading model and tokenizer."""
        if not self._loaded:
            return

        logger.info("Unloading summary model")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._vram_mb = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Summary model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_prompt(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        """Build the summary prompt from frame captions and metadata."""
        duration_str = format_duration(duration if duration else None)
        participants_str = format_participants(performer_count, performer_genders)
        promotional_summary = promo_desc or "Not available"

        frame_lines: List[str] = []
        for frame in frame_captions:
            timestamp = frame.get("timestamp", 0)
            caption = clean_frame_caption(frame.get("caption", ""))
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            frame_lines.append(f"[{mins:02d}:{secs:02d}] {caption}")

        return SUMMARY_PROMPT_TEMPLATE.format(
            duration_str=duration_str,
            frame_count=len(frame_captions),
            resolution=resolution,
            participants_str=participants_str,
            promotional_summary=promotional_summary,
            frame_descriptions="\n".join(frame_lines),
        )

    def generate_summary(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Generate a narrative summary from frame captions and metadata.

        Uses streaming generation so progress can be reported token-by-token.
        The model must be loaded via load() before calling this method.
        Call via asyncio.to_thread() to avoid blocking the event loop.

        Args:
            progress_callback: Optional callable(tokens_generated, max_tokens)
                invoked as tokens stream out. Runs in the calling thread.

        Returns:
            Narrative summary text (2-4 paragraphs).
        """
        if not self._loaded:
            raise RuntimeError("Summary model not loaded. Call load() first.")

        from transformers import TextIteratorStreamer

        prompt = self._build_prompt(
            frame_captions=frame_captions,
            promo_desc=promo_desc,
            duration=duration,
            performer_count=performer_count,
            performer_genders=performer_genders,
            resolution=resolution,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if hasattr(inputs, "input_ids"):
            input_ids = inputs["input_ids"].to(self.model.device)
        elif isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.model.device)
        else:
            input_ids = torch.tensor(inputs, dtype=torch.long).to(self.model.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_len = input_ids.shape[-1]
        logger.debug("Generating summary (%d input tokens)", input_len)

        # Run generate() in a background thread so we can iterate over the
        # streamer in the current thread and fire progress callbacks.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
        )
        generation_kwargs = dict(
            inputs=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        def _run_generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        gen_thread = threading.Thread(target=_run_generate, daemon=True)
        gen_thread.start()

        chunks: List[str] = []
        tokens_generated = 0
        for text_chunk in streamer:
            chunks.append(text_chunk)
            # Approximate token count from whitespace split — close enough
            # for progress reporting, avoids re-tokenizing each chunk.
            tokens_generated += max(1, len(text_chunk.split()))
            if progress_callback is not None:
                try:
                    progress_callback(tokens_generated, self.max_tokens)
                except Exception:
                    logger.debug("progress_callback raised", exc_info=True)

        gen_thread.join()
        summary = "".join(chunks).strip()

        logger.debug("Generated summary (%d chars, ~%d tokens)", len(summary), tokens_generated)
        return summary

    async def is_available(self) -> bool:
        """Check if the summary model can be loaded."""
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            return True
        except Exception:
            return False
