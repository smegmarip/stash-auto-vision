"""
Captioning Service - JoyCaption VLM Processor
Handles model loading, inference, and caption generation using JoyCaption Alpha Two
"""

import os
import gc
import time
from typing import List, Optional, Dict, Any, Tuple
import logging

import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class JoyCaptionProcessor:
    """
    JoyCaption VLM processor for image captioning

    Uses JoyCaption Alpha Two (based on Llama 3.1 8B)
    Supports 4-bit quantization for reduced VRAM usage (~8GB vs ~17GB)
    """

    def __init__(
        self,
        model_name: str = "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        device: str = "cuda",
        use_quantization: bool = True,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize JoyCaption processor

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda, cpu)
            use_quantization: Use 4-bit quantization to reduce VRAM
            max_length: Maximum output token length
            cache_dir: Optional model cache directory
        """
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        self.max_length = max_length
        self.cache_dir = cache_dir or os.getenv("HF_HOME", "/root/.cache/huggingface")

        self.model = None
        self.processor = None
        self.model_loaded = False
        self.vram_usage_mb: Optional[float] = None

    def load_model(self) -> Dict[str, Any]:
        """
        Load JoyCaption model and processor

        Returns:
            Dict with model info (name, device, quantization, vram)
        """
        if self.model_loaded:
            return self.get_model_info()

        logger.info(f"Loading JoyCaption model: {self.model_name}")
        start_time = time.time()

        try:
            from transformers import (
                AutoProcessor,
                LlavaForConditionalGeneration,
                BitsAndBytesConfig
            )

            # Configure quantization
            # llm_int8_skip_modules works for both 8-bit and 4-bit despite its name
            # Include both direct and nested paths for module names
            if self.use_quantization and self.device == "cuda":
                modules_to_skip = [
                    "vision_tower", "multi_modal_projector",
                    "model.vision_tower", "model.multi_modal_projector",
                    "lm_head"  # Also skip the output layer
                ]
                logger.info(f"Using 4-bit quantization, skipping: {modules_to_skip}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=modules_to_skip,
                )
            else:
                quantization_config = None

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            # Load model
            if quantization_config:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                if self.device == "cuda":
                    self.model = self.model.to(self.device)

            self.model.eval()
            self.model_loaded = True

            # Measure VRAM usage
            if self.device == "cuda":
                torch.cuda.synchronize()
                self.vram_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.1f}s, VRAM: {self.vram_usage_mb:.0f}MB")

            return self.get_model_info()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload_model(self):
        """Unload model from memory"""
        if not self.model_loaded:
            return

        logger.info("Unloading JoyCaption model...")

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.model_loaded = False

        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.vram_usage_mb = None
        logger.info("Model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "device": self.device,
            "quantization": "4-bit" if self.use_quantization else "none",
            "loaded": self.model_loaded,
            "vram_mb": self.vram_usage_mb,
            "max_length": self.max_length
        }

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Handle BGR to RGB conversion (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]

        return Image.fromarray(image)

    def caption_image(
        self,
        image: np.ndarray,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate caption for a single image

        Args:
            image: Image as numpy array (H, W, C) in BGR or RGB format
            prompt: Text prompt for captioning style
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated caption string
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)

        # Build conversation - use string content for LLaVA chat template
        conversation = [
            {
                "role": "user",
                "content": f"<image>\n{prompt}"
            }
        ]

        # Apply chat template - ensure we get a string, not a list
        prompt_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Handle case where apply_chat_template still returns a list
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0] if prompt_text else ""

        # Process inputs
        inputs = self.processor(
            text=prompt_text,
            images=pil_image,
            return_tensors="pt"
        )

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # Decode output
        # Skip input tokens to get only generated text
        input_length = inputs["input_ids"].shape[1]
        output_length = output_ids.shape[1]
        generated_ids = output_ids[0, input_length:]

        logger.info(f"Generation: input={input_length}, output={output_length}, new_tokens={len(generated_ids)}")

        # Decode full output first (for debugging)
        full_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        caption = self.processor.decode(generated_ids, skip_special_tokens=True)

        if len(generated_ids) == 0:
            logger.warning(f"No tokens generated! Full output: {full_output[:500]}")
        else:
            logger.info(f"Caption preview: {caption[:100]}...")

        return caption.strip()

    def caption_images_batch(
        self,
        images: List[np.ndarray],
        prompt: str,
        batch_size: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Generate captions for multiple images

        Args:
            images: List of images as numpy arrays
            prompt: Text prompt for captioning style
            batch_size: Number of images to process at once
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of caption strings
        """
        captions = []
        total = len(images)

        for i in range(0, total, batch_size):
            batch = images[i:i + batch_size]

            # Currently process one at a time (batch=1 most efficient for VLM)
            for j, image in enumerate(batch):
                try:
                    caption = self.caption_image(image, prompt, temperature, top_p)
                    captions.append(caption)
                except Exception as e:
                    import traceback
                    logger.error(f"Error captioning image {i + j}: {e}\n{traceback.format_exc()}")
                    captions.append("")

                if progress_callback:
                    progress_callback(i + j + 1, total)

        return captions

    def get_vram_usage(self) -> Tuple[float, float]:
        """
        Get current VRAM usage

        Returns:
            Tuple of (allocated_mb, reserved_mb)
        """
        if self.device != "cuda":
            return (0.0, 0.0)

        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        return (allocated, reserved)


def create_processor(
    device: str = "cuda",
    use_quantization: bool = True
) -> JoyCaptionProcessor:
    """
    Factory function to create JoyCaption processor

    Args:
        device: Device to use
        use_quantization: Whether to use 4-bit quantization

    Returns:
        JoyCaptionProcessor instance (model not yet loaded)
    """
    return JoyCaptionProcessor(
        device=device,
        use_quantization=use_quantization
    )
