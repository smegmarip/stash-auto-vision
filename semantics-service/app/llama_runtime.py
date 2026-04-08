"""
Shared Llama runtime wrapper.

Encapsulates load/unload and chat-style generation for a local Llama model so
multiple consumers (SummaryGenerator, TitleGenerator) can share a single
in-memory copy of the weights. The ModelManager registers this runtime
directly — consumers do not own the model lifecycle.
"""

import gc
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "RedHatAI/Llama-3.1-8B-Instruct"


class LlamaRuntime:
    """Owns a single loaded Llama model and tokenizer.

    Exposes ``load``/``unload``/``is_loaded`` so it can plug into the
    ModelManager, plus ``generate`` for chat-style completion. Consumers
    compose messages and call ``generate`` — they don't touch the model
    or tokenizer directly.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("SEMANTICS_LLM_MODEL", DEFAULT_MODEL)
        self.device = device or os.getenv("SEMANTICS_LLM_DEVICE", "cuda")
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._vram_mb: Optional[float] = None

    def load(self) -> None:
        """Load model and tokenizer into memory.

        On CUDA: uses 8-bit quantization to fit alongside the classifier in
        16GB VRAM. On CPU: uses bfloat16 (native Llama precision).
        """
        if self._loaded:
            return

        logger.info("Loading Llama runtime: %s (device=%s)", self.model_name, self.device)
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
                logger.info("Using 8-bit quantization for Llama runtime")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
            logger.info(
                "Llama runtime loaded in %.1fs (device=%s, VRAM: %.0fMB)",
                elapsed, self.device, self._vram_mb or 0,
            )

        except Exception:
            logger.exception("Failed to load Llama runtime")
            raise

    def unload(self) -> None:
        """Free GPU memory by unloading model and tokenizer."""
        if not self._loaded:
            return

        logger.info("Unloading Llama runtime")
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

        logger.info("Llama runtime unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def is_available(self) -> bool:
        """Check if the underlying model can be loaded."""
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            return True
        except Exception:
            return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Generate a chat completion for the given messages.

        Uses streaming generation so callers can report progress token-by-token.
        The runtime must be loaded via load() before calling this method.
        Call via asyncio.to_thread() to avoid blocking the event loop.

        Args:
            messages: OpenAI-style chat messages [{"role": ..., "content": ...}, ...]
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature (0 disables sampling).
            top_p: Top-p nucleus sampling threshold.
            progress_callback: Optional callable(tokens_generated, max_tokens)
                invoked as tokens stream out. Runs in the calling thread.

        Returns:
            Generated completion text (stripped).
        """
        if not self._loaded:
            raise RuntimeError("Llama runtime not loaded. Call load() first.")

        from transformers import TextIteratorStreamer

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
        logger.debug("Llama generate (%d input tokens, max_new=%d)", input_len, max_tokens)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
        )
        generation_kwargs = dict(
            inputs=input_ids,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
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
                    progress_callback(tokens_generated, max_tokens)
                except Exception:
                    logger.debug("progress_callback raised", exc_info=True)

        gen_thread.join()
        result = "".join(chunks).strip()

        logger.debug("Llama generate complete (%d chars, ~%d tokens)", len(result), tokens_generated)
        return result
