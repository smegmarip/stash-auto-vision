"""
Faces Service - Recognition Manager
Manages multiple InsightFace instances with different det_sizes for optimal detection
"""

import logging
import numpy as np
from typing import Dict, Tuple

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available")

logger = logging.getLogger(__name__)


class RecognitionManager:
    """
    Manages multiple FaceAnalysis instances with different det_sizes

    Strategy:
    - CPU mode: 2 instances (det_size=320, 640)
    - GPU mode: 3 instances (det_size=320, 640, 1024)

    Selection based on input image dimensions:
    - Images <500px min dimension: use det_size=320 (avoid excessive upscaling)
    - Images 500-1500px min dimension: use det_size=640 (standard)
    - Images >1500px min dimension: use det_size=1024 (GPU only, fallback to 640 on CPU)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda"
    ):
        """
        Initialize recognition manager with multiple FaceAnalysis instances

        Args:
            model_name: InsightFace model (buffalo_l, buffalo_s, buffalo_sc)
            device: cuda or cpu
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not installed")

        self.model_name = model_name
        self.device = device
        self.apps: Dict[int, FaceAnalysis] = {}

        # Determine providers and context
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        ctx_id = 0 if device == 'cuda' else -1

        # Load instances based on device
        self._load_instances(model_name, providers, ctx_id)

        logger.info(
            f"RecognitionManager initialized: {model_name} on {device} "
            f"with {len(self.apps)} det_size instances {list(self.apps.keys())}"
        )

    def _load_instances(self, model_name: str, providers: list, ctx_id: int):
        """
        Load FaceAnalysis instances with different det_sizes

        Args:
            model_name: InsightFace model name
            providers: ONNX Runtime providers
            ctx_id: Context ID (0 for GPU, -1 for CPU)
        """
        # Instance 1: Small images (det_size=320)
        logger.info("Loading FaceAnalysis instance: det_size=320 (small images <500px)")
        app_320 = FaceAnalysis(name=model_name, providers=providers)
        app_320.prepare(ctx_id=ctx_id, det_size=(320, 320))
        self.apps[320] = app_320

        # Instance 2: Medium/Large images (det_size=640)
        logger.info("Loading FaceAnalysis instance: det_size=640 (medium/large images 500-1500px)")
        app_640 = FaceAnalysis(name=model_name, providers=providers)
        app_640.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.apps[640] = app_640

        # Instance 3: Very large images (det_size=1024) - GPU only
        if self.device == 'cuda':
            logger.info("Loading FaceAnalysis instance: det_size=1024 (very large images >1500px)")
            app_1024 = FaceAnalysis(name=model_name, providers=providers)
            app_1024.prepare(ctx_id=ctx_id, det_size=(1024, 1024))
            self.apps[1024] = app_1024

    def select_app(self, image: np.ndarray) -> Tuple[FaceAnalysis, int]:
        """
        Select appropriate FaceAnalysis instance based on image dimensions

        Selection strategy:
        1. Calculate minimum dimension of input image
        2. Select det_size to minimize aggressive upscaling/downscaling:
           - <500px: use det_size=320 (avoid 2x+ upscaling)
           - 500-1500px: use det_size=640 (standard range)
           - >1500px: use det_size=1024 if available (GPU), else 640

        Args:
            image: Input image array (H, W, C)

        Returns:
            Tuple of (FaceAnalysis instance, det_size used)
        """
        height, width = image.shape[:2]
        min_dim = min(height, width)

        # Selection logic
        if min_dim < 500:
            selected_size = 320
            reason = f"small image ({width}x{height}, min_dim={min_dim})"
        elif min_dim < 1500 or 1024 not in self.apps:
            selected_size = 640
            if min_dim >= 1500:
                reason = f"large image ({width}x{height}) but GPU instance not available"
            else:
                reason = f"medium image ({width}x{height}, min_dim={min_dim})"
        else:
            selected_size = 1024
            reason = f"very large image ({width}x{height}, min_dim={min_dim})"

        logger.debug(f"Selected det_size={selected_size} for {reason}")

        return self.apps[selected_size], selected_size

    def get_model_info(self) -> Dict:
        """
        Get model configuration information

        Returns:
            Dict with model info
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "det_sizes": list(self.apps.keys()),
            "num_instances": len(self.apps),
            "insightface_available": INSIGHTFACE_AVAILABLE
        }
