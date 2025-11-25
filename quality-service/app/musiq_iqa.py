"""
Faces Service - MUSIQ Image Quality Assessment Module
Wrapper for TensorFlow Hub MUSIQ IQA model
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import logging
import resource

logger = logging.getLogger(__name__)

tf.debugging.set_log_device_placement(True)
tf.config.experimental.enable_op_determinism()


class MusiqIQA:
    """
    Wrapper for TFHub MUSIQ IQA model.
    """

    handles = {
        "spaq": "https://tfhub.dev/google/musiq/spaq/1",
        "koniq-10k": "https://tfhub.dev/google/musiq/koniq-10k/1",
        "paq2piq": "https://tfhub.dev/google/musiq/paq2piq/1",
        "ava": "https://tfhub.dev/google/musiq/ava/1",
    }

    def __init__(self, model_name="koniq-10k"):
        if model_name not in self.handles:
            raise ValueError(f"Unknown MUSIQ model: {model_name}")

        logger.info(f"Loading MUSIQ model: {model_name} ({self.handles[model_name]})")
        self.model = hub.load(self.handles[model_name])
        self.predict_fn = self.model.signatures["serving_default"]

    def score(self, img_bgr):
        """
        Takes BGR uint8 image, returns MUSIQ quality score.
        MUSIQ expects a scalar string tensor encoded as JPEG/PNG.
        """

        self.log_mem("before inference")
        # Encode cropped face
        success, encoded = cv2.imencode(".jpg", img_bgr)
        logger.debug(f"Encoded image shape: {encoded.shape}, success: {success}")
        if not success:
            logger.error("Failed to encode image for MUSIQ scoring")
            return 0.25

        image_bytes = encoded.tobytes()

        # MUSIQ signature: image_bytes_tensor=tf.string
        inputs = {"image_bytes_tensor": tf.convert_to_tensor(image_bytes, dtype=tf.string)}

        result = self._predict(**inputs)
        self.log_mem("after inference")
        return float(result["output_0"].numpy()[0])

    @tf.function(experimental_relax_shapes=True, autograph=False)
    def _predict(self, image_bytes_tensor):
        return self.predict_fn(image_bytes_tensor=image_bytes_tensor)

    def log_mem(self, tag=""):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss / 1024  # ru_maxrss is KB on Linux
        logging.info(f"[MEM {tag}] RSS = {rss_mb:.2f} MB")
