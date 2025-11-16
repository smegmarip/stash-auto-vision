"""
Frame Server - GFPGAN Enhancer
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .base import BaseEnhancer

logger = logging.getLogger(__name__)

# Try importing GFPGAN dependencies
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GFPGAN not available: {e}")
    GFPGAN_AVAILABLE = False
    torch = None
    GFPGANer = None

try:
    from basicsr.utils.download_util import load_file_from_url
    BASICSR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BasicSR utilities not available: {e}")
    BASICSR_AVAILABLE = False


class GFPGANEnhancer(BaseEnhancer):
    """GFPGAN-based face enhancement"""

    def __init__(self, device: str = "cpu", model_dir: str = "/tmp/face_enhancement_models"):
        """
        Initialize GFPGAN enhancer

        Args:
            device: "cuda" or "cpu"
            model_dir: Directory to store downloaded models
        """
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.enhancer = None
        self.bg_upsampler = None

        if not GFPGAN_AVAILABLE:
            logger.error("GFPGAN dependencies not available")
            return

        self._init()

    def _init(self):
        """Initialize GFPGAN model"""
        try:
            logger.info(f"Initializing GFPGAN on {self.device}...")

            # GFPGAN v1.4 model URLs
            model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            model_path = self.model_dir / "GFPGANv1.4.pth"

            # Download model if needed
            if not model_path.exists():
                logger.info(f"Downloading GFPGAN model to {model_path}...")
                if BASICSR_AVAILABLE:
                    load_file_from_url(
                        url=model_url,
                        model_dir=str(self.model_dir),
                        progress=True,
                        file_name="GFPGANv1.4.pth"
                    )
                else:
                    logger.error("Cannot download model - basicsr utilities not available")
                    return

            # Initialize background upsampler (optional)
            bg_model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            bg_model_path = self.model_dir / "RealESRGAN_x2plus.pth"

            if not bg_model_path.exists():
                logger.info("Downloading background upsampler...")
                if BASICSR_AVAILABLE:
                    load_file_from_url(
                        url=bg_model_url,
                        model_dir=str(self.model_dir),
                        progress=True,
                        file_name="RealESRGAN_x2plus.pth"
                    )

            # Create background upsampler
            if bg_model_path.exists():
                from realesrgan import RealESRGANer
                bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                self.bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path=str(bg_model_path),
                    model=bg_model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False if self.device == "cpu" else True,
                    device=self.device
                )
                logger.info("Background upsampler initialized")
            else:
                logger.warning("Background upsampler not available - faces only")

            # Create GFPGAN enhancer
            self.enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.bg_upsampler,
                device=self.device
            )

            logger.info("GFPGAN initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize GFPGAN: {e}", exc_info=True)
            self.enhancer = None

    def is_available(self) -> bool:
        """Check if GFPGAN is initialized"""
        return self.enhancer is not None

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
        Enhance faces using GFPGAN

        Args:
            frame: Input frame (BGR format)
            fidelity_weight: Balance between quality and fidelity (0.0-1.0)
            upscale: Upscaling factor (ignored for GFPGAN, always 2x)
            only_center_face: Only enhance the most centered face
            aligned: Whether input face is aligned
            paste_back: Paste enhanced faces back to original image

        Returns:
            Tuple of (enhanced_frame, num_faces_enhanced)
        """
        if not self.is_available():
            logger.warning("GFPGAN not available - returning original frame")
            return frame, 0

        try:
            # GFPGAN expects BGR input (OpenCV format)
            _, _, restored_img = self.enhancer.enhance(
                frame,
                has_aligned=aligned,
                only_center_face=only_center_face,
                paste_back=paste_back,
                weight=fidelity_weight
            )

            if restored_img is None:
                logger.warning("Enhancement returned None - using original frame")
                return frame, 0

            num_faces = 1 if restored_img is not None else 0

            return restored_img, num_faces

        except Exception as e:
            logger.error(f"GFPGAN enhancement failed: {e}", exc_info=True)
            return frame, 0

    def cleanup(self):
        """Cleanup GFPGAN resources"""
        if self.enhancer is not None:
            del self.enhancer
            self.enhancer = None

        if self.bg_upsampler is not None:
            del self.bg_upsampler
            self.bg_upsampler = None

        if torch is not None and self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("GFPGAN cleanup complete")

    def __del__(self):
        """Destructor"""
        self.cleanup()
