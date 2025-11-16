"""
Frame Server - CodeFormer Enhancer
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .base import BaseEnhancer

logger = logging.getLogger(__name__)

# Try importing CodeFormer dependencies
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None

try:
    from basicsr.utils.download_util import load_file_from_url
    BASICSR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BasicSR utilities not available: {e}")
    BASICSR_AVAILABLE = False

try:
    from codeformer.basicsr.archs.codeformer_arch import CodeFormer
    from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
    CODEFORMER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CodeFormer not available: {e}")
    CODEFORMER_AVAILABLE = False
    CodeFormer = None
    FaceRestoreHelper = None


class CodeFormerEnhancer(BaseEnhancer):
    """CodeFormer-based face enhancement"""

    def __init__(self, device: str = "cpu", model_dir: str = "/tmp/face_enhancement_models"):
        """
        Initialize CodeFormer enhancer

        Args:
            device: "cuda" or "cpu"
            model_dir: Directory to store downloaded models
        """
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.face_helper = None
        self.bg_upsampler = None

        if not CODEFORMER_AVAILABLE:
            logger.error("CodeFormer dependencies not available")
            return

        self._init()

    def _init(self):
        """Initialize CodeFormer model"""
        try:
            logger.info(f"Initializing CodeFormer on {self.device}...")

            # CodeFormer model URL
            model_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            model_path = self.model_dir / "codeformer.pth"

            # Download model if needed
            if not model_path.exists():
                logger.info(f"Downloading CodeFormer model to {model_path}...")
                if BASICSR_AVAILABLE:
                    load_file_from_url(
                        url=model_url,
                        model_dir=str(self.model_dir),
                        progress=True,
                        file_name="codeformer.pth"
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

            # Initialize CodeFormer network
            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256']
            ).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(str(model_path), map_location=self.device)
            net.load_state_dict(checkpoint['params_ema'])
            net.eval()

            # Create face restoration helper
            self.face_helper = FaceRestoreHelper(
                upscale_factor=2,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device
            )
            self.face_helper.net = net
            self.face_helper.bg_upsampler = self.bg_upsampler

            logger.info("CodeFormer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CodeFormer: {e}", exc_info=True)
            self.face_helper = None

    def is_available(self) -> bool:
        """Check if CodeFormer is initialized"""
        return self.face_helper is not None

    def enhance_frame(
        self,
        frame: np.ndarray,
        fidelity_weight: float = 0.7,
        upscale: int = 2,
        only_center_face: bool = False,
        aligned: bool = False,
        paste_back: bool = True,
        face_upsample: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Enhance faces using CodeFormer

        Args:
            frame: Input frame (BGR format)
            fidelity_weight: Balance between quality and fidelity (0.0-1.0)
            upscale: Upscaling factor (ignored for CodeFormer, always 2x)
            only_center_face: Only enhance the most centered face
            aligned: Whether input face is aligned (ignored for CodeFormer)
            paste_back: Paste enhanced faces back to original image
            face_upsample: Apply additional upsampling to restored faces (default True)

        Returns:
            Tuple of (enhanced_frame, num_faces_enhanced)
        """
        if not self.is_available():
            logger.warning("CodeFormer not available - returning original frame")
            return frame, 0

        try:
            # CodeFormer uses FaceRestoreHelper
            self.face_helper.clean_all()
            self.face_helper.read_image(frame)

            # Detect faces
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face)
            self.face_helper.align_warp_face()

            # Enhance faces
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                # Convert BGR to RGB and normalize to [0, 1]
                cropped_face_t = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB) / 255.0
                # Convert to tensor (H, W, C) -> (C, H, W)
                cropped_face_t = torch.from_numpy(cropped_face_t.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                # Normalize to [-1, 1] range (mean=0.5, std=0.5) as expected by CodeFormer
                cropped_face_t = (cropped_face_t - 0.5) / 0.5

                with torch.no_grad():
                    output = self.face_helper.net(cropped_face_t, w=fidelity_weight)
                    # Output is a tuple (output_tensor, code_tensor), we want the output_tensor
                    if isinstance(output, tuple):
                        output = output[0]
                    # Denormalize from [-1, 1] to [0, 1]
                    output = output * 0.5 + 0.5
                    restored_face = torch.clamp(output, 0, 1).cpu().numpy()

                # restored_face shape: (1, C, H, W) -> need (H, W, C)
                if restored_face.ndim == 4:  # (1, C, H, W)
                    restored_face = restored_face.squeeze(0)  # (C, H, W)
                restored_face = restored_face.transpose(1, 2, 0)  # (H, W, C)
                restored_face = cv2.cvtColor((restored_face * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                self.face_helper.add_restored_face(restored_face)

            num_faces = len(self.face_helper.cropped_faces)

            # Paste faces back if requested
            if paste_back and num_faces > 0:
                self.face_helper.get_inverse_affine(None)
                upsample_img = self.bg_upsampler.enhance(frame)[0] if self.bg_upsampler else None
                # Apply face upsampler if requested (uses bg_upsampler for face regions)
                face_upsampler = self.bg_upsampler if face_upsample else None
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=upsample_img,
                    face_upsampler=face_upsampler
                )
            else:
                restored_img = frame

            return restored_img, num_faces

        except Exception as e:
            logger.error(f"CodeFormer enhancement failed: {e}", exc_info=True)
            return frame, 0

    def cleanup(self):
        """Cleanup CodeFormer resources"""
        if self.face_helper is not None:
            del self.face_helper
            self.face_helper = None

        if self.bg_upsampler is not None:
            del self.bg_upsampler
            self.bg_upsampler = None

        if torch is not None and self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("CodeFormer cleanup complete")

    def __del__(self):
        """Destructor"""
        self.cleanup()
