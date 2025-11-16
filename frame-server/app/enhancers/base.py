"""
Frame Server - Base Enhancer Interface
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseEnhancer(ABC):
    """Abstract base class for face enhancement models"""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the enhancer is initialized and ready"""
        pass

    @abstractmethod
    def enhance_frame(
        self,
        frame: np.ndarray,
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
            fidelity_weight: Balance between quality and fidelity (0.0-1.0)
            upscale: Upscaling factor (1, 2, 3, 4)
            only_center_face: Only enhance the most centered face
            aligned: Whether input face is aligned
            paste_back: Paste enhanced faces back to original image

        Returns:
            Tuple of (enhanced_frame, num_faces_enhanced)
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
