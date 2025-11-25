"""
Faces Service - Face Quality Assessment Module
Assesses face quality using bbox size, pose, occlusion, and sharpness
"""

import os
import logging
import numpy as np
import cv2

from .iqa_onnx import IQAModel

IQA_MODEL = os.getenv("IQA_MODEL", "topiq")

logger = logging.getLogger(__name__)


class FaceQuality:
    """
    Face quality assessment class

    Evaluates face quality based on bounding box size, pose, occlusion, and sharpness.
    """

    def __init__(self):
        self.model_name = IQA_MODEL
        self.iqa = IQAModel(preferred_model=self.model_name)
        logger.info(f"Face quality assessment initialized with IQA model: {self.iqa.model_type}")

    def _interp(self, x, x0, x1, y0, y1):
        """
        Generic smooth interpolation utility
        Args:
            x: Input value
            x0: Lower bound of input
            x1: Upper bound of input
            y0: Output value at lower bound
            y1: Output value at upper bound
        Returns:
            Interpolated output value
        """
        if x <= x0:
            # extrapolate, but never return < 0
            return max(y0 - (x0 - x) * (y1 - y0) / (x1 - x0), 0.0)
        if x >= x1:
            return y1
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def _size_score(self, face_min_dim):
        """
        Calculate size score based on the minimum dimension of the face bounding box.

        Args:
            face_min_dim: Minimum dimension of the face bounding box (in pixels)

        Returns:
            Size score between 80 → 0.25 (barely usable) and 250 → 1.0 (excellent)
        """
        return self._interp(face_min_dim, 80, 250, 0.25, 1.0)

    def _estimate_pose_angles(self, face):
        """
        Estimate yaw, pitch, and roll angles from facial landmarks.

        Args:
            face: Face object with facial landmarks
        Returns:
            yaw, pitch, roll angles in degrees
        """
        try:
            # Use native InsightFace pose angles if available
            if hasattr(face, "pose"):
                pitch, yaw, roll = face.pose  # [pitch, yaw, roll] in degrees
                return yaw, pitch, roll
        except Exception as e:
            logger.warning(f"Failed to get pose from face object: {e}")

        # Fallback: estimate pose from landmarks
        # Get landmarks
        if hasattr(face, "kps"):
            try:
                kps = face.kps

                # Calculate eye center and mouth center
                left_eye = kps[0]
                right_eye = kps[1]
                mouth_center = (kps[3] + kps[4]) / 2

                # Calculate eye distance and vertical alignment
                eye_center = (left_eye + right_eye) / 2
                eye_distance = np.linalg.norm(right_eye - left_eye)

                # Calculate rotation angle
                eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                angle_degrees = np.degrees(eye_angle)

                # Determine pose
                if abs(angle_degrees) < 15:
                    # Check for left/right turn
                    nose = kps[2]
                    eye_to_nose_x = nose[0] - eye_center[0]

                    # Estimate roll from eye angle
                    roll = angle_degrees

                    # Estimate yaw from horizontal asymmetry
                    # When face turns right, nose moves right relative to eye center
                    yaw = np.degrees(np.arctan2(eye_to_nose_x, eye_distance)) * 2.0

                    # Estimate pitch from vertical face proportions
                    # When looking up, eyes appear higher relative to mouth
                    face_height = np.linalg.norm(mouth_center - eye_center)
                    expected_ratio = 0.7  # Typical eye-to-mouth / eye-distance ratio
                    actual_ratio = face_height / eye_distance
                    pitch = (actual_ratio - expected_ratio) * 50.0  # Scale to degrees
                    return yaw, pitch, roll
            except Exception as e:
                logger.warning(f"Failed to estimate pose from landmarks: {e}")
                return 0.0, 0.0, 0.0
        return 0.0, 0.0, 0.0

    def _pose_score(self, yaw, pitch):
        """
        Calculate pose score based on yaw and pitch angles.

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
        """
        yaw_score = self._interp(abs(yaw), 0, 45, 1.0, 0.25)
        pitch_score = self._interp(abs(pitch), 0, 35, 1.0, 0.25)

        return (yaw_score + pitch_score) / 2

    def _occlusion_score(self, occlusion_pred, occlusion_prob):
        """
        Calculate occlusion score based on occlusion prediction and probability.

        Args:
            occlusion_pred: Occlusion prediction (not used in current calculation)
            occlusion_prob: Occlusion probability (0.0 to 1.0)

        Returns:
            Occlusion score between 1.0 (no occlusion) and 0.25 (high occlusion)
        """
        # Convert (occlusion_pred, occlusion_prob) to occlusion metric
        if occlusion_pred == 0:  # non-occluded winner
            occlusion_metric = occlusion_prob  # 0→1 as confidence increases
        else:  # occluded winner
            occlusion_metric = -occlusion_prob  # 0→-1 as confidence increases

        # Map from [-1, 1] to [0.25, 1.0]
        # -1 (certain occluded) → 0.25
        # +1 (certain non-occluded) → 1.0
        return self._interp(occlusion_metric, -1.0, 1.0, 0.25, 1.0)

    def _sharpness_score(self, face_img):
        """
        Calculate sharpness score using ONNX IQA models (TOPIQ/CLIP-IQA/Sobel).
        Args:
            face_img: Cropped face image (H, W, 3) BGR format
        Returns:
            Sharpness score between 0.0 and 1.0 (higher = better quality)
        """
        try:
            # IQA models (TOPIQ, CLIP-IQA, Sobel) return scores in [0, 1] range
            score = self.iqa.score(face_img)
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Failed to calculate IQA sharpness score: {e}")
            # Return neutral quality if scoring fails completely
            return 0.5

    def calculate_quality(self, face, frame, occlusion_data):
        """
        Calculate overall face quality score with component breakdown.
        Args:
            face: Face object with bbox and other attributes
            frame: Original image frame (H, W, 3) BGR format
            occlusion_data: Tuple of (occlusion_pred, occlusion_prob)
        Returns:
            Dict with composite score and component scores:
            {
                'composite': 0.907,
                'components': {
                    'size': 0.85,
                    'pose': 0.92,
                    'occlusion': 0.95,
                    'sharpness': 0.88
                }
            }
        """
        ih, iw = frame.shape[0:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        y1, y2 = max(0, y1), min(ih, y2)
        x1, x2 = max(0, x1), min(iw, x2)
        h = y2 - y1
        w = x2 - x1
        face_min_dim = min(h, w)
        (occlusion_pred, occlusion_prob) = occlusion_data

        size_s = self._size_score(face_min_dim)
        yaw, pitch, _ = self._estimate_pose_angles(face)
        pose_s = self._pose_score(yaw, pitch)
        occ_s = self._occlusion_score(occlusion_pred, occlusion_prob)

        face_img = frame[y1:y2, x1:x2]
        sharp_s = self._sharpness_score(face_img) if face_img.size else 0.25

        composite = 0.35 * size_s + 0.20 * pose_s + 0.20 * occ_s + 0.25 * sharp_s
        composite = float(min(max(composite, 0.0), 1.0))

        logger.debug(f"Face quality: composite={composite:.3f}, size={size_s:.3f}, pose={pose_s:.3f}, occlusion={occ_s:.3f}, sharpness={sharp_s:.3f}")

        return {
            'composite': composite,
            'components': {
                'size': float(size_s),
                'pose': float(pose_s),
                'occlusion': float(occ_s),
                'sharpness': float(sharp_s)
            }
        }
