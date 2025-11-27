"""
ONNX IQA Module
Laplacian variance for face sharpness scoring.
TOPIQ-NR and CLIP-IQA+ ONNX models retained for potential future use
(e.g., semantics or object detection services).
"""

import os
import logging
import numpy as np
import cv2
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# Environment configuration
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")


class IQAModel:
    """
    Face Sharpness Assessment using Laplacian variance.

    Laplacian variance measures actual edge sharpness/blur, which is
    appropriate for face crops. TOPIQ/CLIP-IQA are MOS-based image quality
    models that don't discriminate well on face crops.

    ONNX model loading code retained for potential future use in other services.
    """

    def __init__(self, preferred_model: str = "laplacian"):
        """
        Initialize sharpness model.

        Args:
            preferred_model: Only "laplacian" is supported for face sharpness.
                            ONNX models (topiq, clipiqa) are not suitable for face crops.
        """
        self.model_type = "laplacian"
        self.session = None
        logger.info("Using Laplacian variance for face sharpness scoring")

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

    def _laplacian_score(self, img_bgr: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian variance.
        Higher variance = sharper edges = better focus.

        Args:
            img_bgr: Input image in BGR format

        Returns:
            Sharpness score [0, 1] (higher = sharper)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Resize to reasonable size for consistency
        target_size = 128
        if min(gray.shape) > target_size:
            scale = target_size / min(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to [0, 1] - typical face range 50-250
        # 50 = very blurry, 250 = very sharp
        score = np.clip((variance - 50) / (250 - 50), 0.0, 1.0)

        return float(score)

    def score(self, img_bgr: np.ndarray) -> float:
        """
        Calculate face sharpness score using Laplacian variance.

        Args:
            img_bgr: Input image in BGR format (OpenCV default)

        Returns:
            Sharpness score [0, 1] (higher = sharper)
        """
        return self._laplacian_score(img_bgr)
