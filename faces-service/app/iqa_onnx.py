"""
ONNX IQA Module
Wrapper for TOPIQ-NR and CLIP-IQA+ ONNX models with Sobel fallback
"""

import os
import logging
import numpy as np
import cv2
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Environment configuration
IQA_MODEL = os.getenv("IQA_MODEL", "topiq")  # topiq, clipiqa, sobel
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")


class IQAModel:
    """
    Image Quality Assessment using ONNX models

    Supports 3-tier fallback:
    1. TOPIQ-NR (most accurate, 384x384, ~133ms)
    2. CLIP-IQA+ (faster, 224x224, ~24ms)
    3. Sobel edge detection (always available)
    """

    def __init__(self, preferred_model: str = "topiq"):
        """
        Initialize IQA model with fallback strategy

        Args:
            preferred_model: "topiq", "clipiqa", or "sobel"
        """
        self.model = None
        self.model_type = None
        self.session = None

        # Try to load preferred model
        if preferred_model == "topiq":
            if self._load_topiq():
                return
            logger.warning("TOPIQ model unavailable, falling back to CLIP-IQA+")
            if self._load_clipiqa():
                return
            logger.warning("CLIP-IQA+ unavailable, falling back to Sobel")
            self._use_sobel()

        elif preferred_model == "clipiqa":
            if self._load_clipiqa():
                return
            logger.warning("CLIP-IQA+ unavailable, falling back to Sobel")
            self._use_sobel()

        else:  # sobel or unknown
            self._use_sobel()

    def _load_topiq(self) -> bool:
        """Load TOPIQ-NR ONNX model"""
        try:
            import onnxruntime as ort

            model_path = os.path.join(MODEL_DIR, "topiq_nr.onnx")
            if not os.path.exists(model_path):
                logger.debug(f"TOPIQ model not found: {model_path}")
                return False

            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.model_type = "topiq"
            logger.info("TOPIQ-NR model loaded successfully (384x384, ~133ms/image)")
            return True

        except Exception as e:
            logger.warning(f"Failed to load TOPIQ model: {e}")
            return False

    def _load_clipiqa(self) -> bool:
        """Load CLIP-IQA+ ONNX model"""
        try:
            import onnxruntime as ort

            model_path = os.path.join(MODEL_DIR, "clipiqa_plus.onnx")
            if not os.path.exists(model_path):
                logger.debug(f"CLIP-IQA+ model not found: {model_path}")
                return False

            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.model_type = "clipiqa"
            logger.info("CLIP-IQA+ model loaded successfully (224x224, ~24ms/image)")
            return True

        except Exception as e:
            logger.warning(f"Failed to load CLIP-IQA+ model: {e}")
            return False

    def _use_sobel(self):
        """Use Sobel edge detection fallback"""
        self.model_type = "sobel"
        logger.info("Using Sobel edge detection for sharpness scoring")

    def _preprocess_image(self, img_bgr: np.ndarray, target_size: int) -> np.ndarray:
        """
        Preprocess image for ONNX model

        Args:
            img_bgr: Input image in BGR format (OpenCV default)
            target_size: Target size (384 for TOPIQ, 224 for CLIP-IQA)

        Returns:
            Preprocessed image [1, 3, H, W] in RGB format, normalized to [0, 1]
        """
        # Convert BGR to RGB (CRITICAL!)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Convert HWC to CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch

    def _sobel_score(self, img_bgr: np.ndarray) -> float:
        """
        Calculate sharpness using Sobel edge detection

        Args:
            img_bgr: Input image in BGR format

        Returns:
            Sharpness score [0, 1] (higher = sharper)
        """
        # Resize to small size for efficiency
        small = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Calculate Sobel gradients
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag = np.sqrt(sx * sx + sy * sy)

        # Mean gradient magnitude
        sharp_feat = mag.mean()

        # Map to [0, 1] using typical face ranges (5-40)
        # 5 = very blurry, 40 = very sharp
        score = np.clip((sharp_feat - 5) / (40 - 5), 0.0, 1.0)

        return float(score)

    def score(self, img_bgr: np.ndarray) -> float:
        """
        Calculate image quality score

        Args:
            img_bgr: Input image in BGR format (OpenCV default)

        Returns:
            Quality score [0, 1] (higher = better quality)
        """
        try:
            if self.model_type == "topiq":
                # TOPIQ-NR: 384x384
                img_input = self._preprocess_image(img_bgr, 384)
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name

                result = self.session.run([output_name], {input_name: img_input})
                score = float(result[0][0][0])

                # TOPIQ outputs are already in [0, 1] range
                return np.clip(score, 0.0, 1.0)

            elif self.model_type == "clipiqa":
                # CLIP-IQA+: 224x224
                img_input = self._preprocess_image(img_bgr, 224)
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name

                result = self.session.run([output_name], {input_name: img_input})
                score = float(result[0][0][0])

                # CLIP-IQA outputs are already in [0, 1] range
                return np.clip(score, 0.0, 1.0)

            else:  # sobel
                return self._sobel_score(img_bgr)

        except Exception as e:
            logger.warning(f"IQA scoring failed ({self.model_type}): {e}, falling back to Sobel")
            return self._sobel_score(img_bgr)
