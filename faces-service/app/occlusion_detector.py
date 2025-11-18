"""
Faces Service - Occlusion Detection Module
Detects occluded faces using LamKser face-occlusion-classification model
"""

import os
import logging
import numpy as np
import cv2
import onnxruntime as ort

logger = logging.getLogger(__name__)


class OcclusionDetector:
    """
    Face occlusion detector using ONNX model

    Detects if a face is occluded (by glasses, mask, hand, etc.)
    Returns both boolean classification and probability score.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize occlusion detector

        Args:
            model_path: Path to ONNX model file (default: models/occlusion_classifier.onnx)
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "occlusion_classifier.onnx"
            )

        self.model_path = model_path
        self.input_size = (224, 224)  # Standard input size for the model

        # Get device from environment (same as face_recognizer)
        device = os.environ.get('INSIGHTFACE_DEVICE', 'cpu').lower()

        # Configure ONNX Runtime providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"Occlusion detector initialized: {model_path} (device: {device}, providers: {providers})")
        except Exception as e:
            logger.error(f"Failed to load occlusion model from {model_path}: {e}")
            logger.warning("Occlusion detection will be disabled")
            self.session = None

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop for model inference

        Args:
            face_crop: Face image crop (H, W, 3) BGR format

        Returns:
            Preprocessed image (1, 3, 224, 224) RGB format, normalized
        """
        # Resize to model input size
        img = cv2.resize(face_crop, self.input_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Normalize with ImageNet stats (standard for most vision models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Transpose to (C, H, W) and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def detect(self, face_crop: np.ndarray) -> tuple[bool, float]:
        """
        Detect if face is occluded

        Args:
            face_crop: Face image crop (H, W, 3) BGR format

        Returns:
            Tuple of (is_occluded, occlusion_probability)
            - is_occluded: True if model predicts occluded class
            - occlusion_probability: 0.0-1.0 probability that face is occluded
        """
        # If model failed to load, return default values
        if self.session is None:
            logger.debug("Occlusion detection skipped (model not loaded)")
            return False, 0.0

        try:
            # Preprocess image
            input_data = self.preprocess(face_crop)

            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )

            # Apply softmax to get probabilities for both classes
            # Following reference implementation: onnx/run_onnx.py
            logits = outputs[0][0]  # Shape: (2,) - [non-occluded, occluded]

            # Softmax: exp(x) / sum(exp(x))
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            probabilities = exp_logits / np.sum(exp_logits)

            # Get prediction using argmax (0=not occluded, 1=occluded)
            pred = np.argmax(probabilities)

            # Extract occlusion probability (class 1)
            occlusion_probability = float(probabilities[1])

            # Binary classification based on model's prediction
            is_occluded = (pred == 1)

            logger.debug(f"Occlusion detection: {is_occluded} (probability: {occlusion_probability:.3f})")

            return is_occluded, occlusion_probability

        except Exception as e:
            logger.error(f"Error during occlusion detection: {e}", exc_info=True)
            return False, 0.0
