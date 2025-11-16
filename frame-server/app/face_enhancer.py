"""
Frame Server - Face Enhancer Factory
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple

from .enhancers.base import BaseEnhancer
from .enhancers.gfpgan_enhancer import GFPGANEnhancer
from .enhancers.codeformer_enhancer import CodeFormerEnhancer

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """Face enhancement factory that wraps different enhancement models"""

    def __init__(
        self,
        model_name: str = "gfpgan",
        device: str = "cpu",
        model_dir: str = "/tmp/face_enhancement_models"
    ):
        """
        Initialize face enhancer with lazy model loading

        Args:
            model_name: Default model ("gfpgan" or "codeformer")
            device: "cuda" or "cpu"
            model_dir: Directory to store downloaded models
        """
        self.default_model_name = model_name
        self.device = device
        self.model_dir = model_dir

        # Cache for lazy-loaded models
        self.models: Dict[str, Optional[BaseEnhancer]] = {}

        # Initialize default model
        self._get_or_create_model(model_name)

    def _get_or_create_model(self, model_name: str) -> Optional[BaseEnhancer]:
        """
        Get or create a model instance (lazy loading with caching)

        Args:
            model_name: "gfpgan" or "codeformer"

        Returns:
            BaseEnhancer instance or None if model unknown/failed
        """
        # Return cached model if available
        if model_name in self.models:
            return self.models[model_name]

        # Create new model instance
        logger.info(f"Initializing {model_name} enhancer...")
        enhancer: Optional[BaseEnhancer] = None

        if model_name == "gfpgan":
            enhancer = GFPGANEnhancer(device=self.device, model_dir=self.model_dir)
        elif model_name == "codeformer":
            enhancer = CodeFormerEnhancer(device=self.device, model_dir=self.model_dir)
        else:
            logger.error(f"Unknown model: {model_name}")

        # Cache the result (even if None)
        self.models[model_name] = enhancer
        return enhancer

    def is_available(self, model_name: Optional[str] = None) -> bool:
        """
        Check if enhancement is available for a specific model

        Args:
            model_name: Model to check, defaults to default_model_name
        """
        model_name = model_name or self.default_model_name
        enhancer = self._get_or_create_model(model_name)
        return enhancer is not None and enhancer.is_available()

    def enhance_frame(
        self,
        frame: np.ndarray,
        model: Optional[str] = None,
        fidelity_weight: float = 0.7,
        upscale: int = 2,
        only_center_face: bool = False,
        aligned: bool = False,
        paste_back: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Enhance faces in a frame

        Args:
            frame: Input frame (BGR format from OpenCV)
            model: Enhancement model to use ("gfpgan" or "codeformer"), defaults to default_model_name
            fidelity_weight: Balance between quality and fidelity (0.0-1.0)
            upscale: Upscaling factor (1, 2, 3, 4)
            only_center_face: Only enhance the most centered face
            aligned: Whether input face is aligned
            paste_back: Paste enhanced faces back to original image

        Returns:
            Tuple of (enhanced_frame, num_faces_enhanced)
        """
        model_name = model or self.default_model_name
        enhancer = self._get_or_create_model(model_name)

        if enhancer is None or not enhancer.is_available():
            logger.warning(f"Enhancement not available for model {model_name} - returning original frame")
            return frame, 0

        return enhancer.enhance_frame(
            frame=frame,
            fidelity_weight=fidelity_weight,
            upscale=upscale,
            only_center_face=only_center_face,
            aligned=aligned,
            paste_back=paste_back
        )

    def enhance_face_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        fidelity_weight: float = 0.7
    ) -> np.ndarray:
        """
        Enhance a specific face region in a frame

        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h)
            fidelity_weight: Enhancement fidelity (0.0-1.0)

        Returns:
            Enhanced frame with face region replaced
        """
        if not self.is_available():
            return frame

        try:
            x, y, w, h = bbox

            # Extract face region with padding
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_region = frame[y1:y2, x1:x2].copy()

            # Enhance face region
            enhanced_region, _ = self.enhance_frame(
                face_region,
                fidelity_weight=fidelity_weight,
                only_center_face=True,
                paste_back=True
            )

            # Paste back into frame
            result = frame.copy()
            result[y1:y2, x1:x2] = enhanced_region

            return result

        except Exception as e:
            logger.error(f"Face region enhancement failed: {e}", exc_info=True)
            return frame

    def cleanup(self):
        """Cleanup resources"""
        if self.enhancer is not None:
            self.enhancer.cleanup()
            self.enhancer = None

        logger.info("Face enhancer cleanup complete")

    def __del__(self):
        """Destructor"""
        self.cleanup()
