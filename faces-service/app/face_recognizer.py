"""
Faces Service - Face Recognizer
InsightFace integration for face detection and recognition
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .frame_client import FrameServerClient

from .occlusion_detector import OcclusionDetector
from .face_quality import FaceQuality
from .recognition_manager import RecognitionManager

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available - face recognition will not work")

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face detection and recognition using InsightFace"""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
        recognition_manager: Optional[RecognitionManager] = None
    ):
        """
        Initialize face recognizer

        Args:
            model_name: InsightFace model (buffalo_l, buffalo_s, buffalo_sc)
            device: cuda or cpu
            det_size: Detection size (width, height) - used if recognition_manager not provided
            recognition_manager: Optional RecognitionManager for multi-size support
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not installed")

        self.model_name = model_name
        self.device = device
        self.det_size = det_size
        self.recognition_manager = recognition_manager

        # If recognition_manager provided, use it; otherwise create single instance
        if self.recognition_manager:
            self.app = None  # Will be selected per-image
            logger.info(f"FaceRecognizer using RecognitionManager with {len(recognition_manager.apps)} instances")
        else:
            # Initialize single InsightFace instance (backward compatibility)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)
            logger.info(f"InsightFace initialized: {model_name} on {device} with det_size={det_size}")

        # Initialize occlusion detector
        self.occlusion_detector = OcclusionDetector()

        # Initialize face quality assessor
        self.face_quality = FaceQuality()

    async def detect_faces(
        self,
        image: np.ndarray,
        face_min_confidence: float = 0.9
    ) -> List[Dict]:
        """
        Detect faces in image

        Args:
            image: Image array (BGR format from OpenCV)
            face_min_confidence: Minimum detection confidence

        Returns:
            List of face detection dicts
        """
        try:
            # Select appropriate FaceAnalysis instance
            if self.recognition_manager:
                app, det_size = self.recognition_manager.select_app(image)
            else:
                app = self.app
                det_size = self.det_size

            # Detect faces
            faces = app.get(image)

            logger.debug(f"InsightFace raw detection: found {len(faces)} faces (det_size={det_size}, face_min_confidence={face_min_confidence})")
            for i, face in enumerate(faces):
                logger.debug(f"  Face {i}: confidence={face.det_score:.3f}")

            # Filter by confidence and convert to dict format
            results = []

            for face in faces:
                if face.det_score < face_min_confidence:
                    logger.debug(f"Skipping face with confidence {face.det_score:.3f} < {face_min_confidence}")
                    continue

                # Get bounding box
                bbox = face.bbox.astype(int)

                # Get landmarks
                landmarks = face.kps.astype(int)

                # Get embedding
                embedding = face.normed_embedding.tolist()

                # Estimate pose
                pose = self._estimate_pose(face)

                # Detect occlusion
                face_crop = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                occlusion_pred, occlusion_prob = self.occlusion_detector.detect(face_crop)

                # Calculate quality score with components using local FaceQuality
                quality_data = self.face_quality.calculate_quality(
                    face=face,
                    frame=image,
                    occlusion_data=(occlusion_pred, occlusion_prob)
                )

                detection = {
                    'bbox': {
                        'x_min': int(bbox[0]),
                        'y_min': int(bbox[1]),
                        'x_max': int(bbox[2]),
                        'y_max': int(bbox[3])
                    },
                    'landmarks': {
                        'left_eye': landmarks[0].tolist(),
                        'right_eye': landmarks[1].tolist(),
                        'nose': landmarks[2].tolist(),
                        'mouth_left': landmarks[3].tolist(),
                        'mouth_right': landmarks[4].tolist()
                    },
                    'embedding': embedding,
                    'confidence': float(face.det_score),
                    'quality': quality_data,
                    'pose': pose,
                    'demographics': None,
                    'enhanced': False,  # Default to False, will be set to True if enhanced
                    'occlusion': {
                        'occluded': bool(occlusion_pred),
                        'probability': float(occlusion_prob)
                    }
                }

                # Add demographics if available
                if hasattr(face, 'age') and hasattr(face, 'gender'):
                    detection['demographics'] = {
                        'age': int(face.age),
                        'gender': 'M' if face.gender == 1 else 'F',
                        'emotion': 'neutral'  # InsightFace doesn't provide emotion by default
                    }

                results.append(detection)

            logger.debug(f"Detected {len(results)} faces (filtered from {len(faces)} total)")
            return results

        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    async def detect_and_enhance_faces(
        self,
        image: np.ndarray,
        video_path: str,
        timestamp: float,
        frame_client: 'FrameServerClient',
        face_min_confidence: float = 0.9,
        face_min_quality: float = 0.0,
        enhancement_enabled: bool = False,
        enhancement_quality_trigger: float = 0.5,
        enhancement_model: str = "codeformer",
        enhancement_fidelity_weight: float = 0.5
    ) -> List[Dict]:
        """
        Detect faces with optional enhancement for low-quality detections

        Args:
            image: Original image array (BGR format from OpenCV)
            video_path: Path to video file (for enhancement)
            timestamp: Timestamp in seconds (for enhancement)
            frame_client: Frame server client for enhancement
            face_min_confidence: Minimum detection confidence
            face_min_quality: Minimum quality threshold (filter below this)
            enhancement_enabled: Enable face enhancement
            enhancement_quality_trigger: Trigger enhancement if quality < this
            enhancement_model: Enhancement model ('gfpgan' or 'codeformer')
            enhancement_fidelity_weight: Fidelity vs quality tradeoff

        Returns:
            List of face detection dicts (after filtering)
        """
        try:
            # First pass: detect faces in original image
            detections = await self.detect_faces(image, face_min_confidence)

            if not enhancement_enabled:
                # No enhancement - just filter by quality
                filtered = [d for d in detections if d['quality']['composite'] >= face_min_quality]
                logger.info(f"No enhancement: {len(filtered)}/{len(detections)} faces pass quality threshold {face_min_quality}")
                return filtered

            # Enhancement workflow
            needs_enhancement = []
            final_detections = []

            for detection in detections:
                # Enhancement is ONLY for high-confidence, low-quality faces
                # This is a corner case: we're confident it's a face, but quality is poor
                # Use face_min_confidence as both the detection AND enhancement threshold
                if (detection['confidence'] >= face_min_confidence and
                    detection['quality']['composite'] < enhancement_quality_trigger):
                    needs_enhancement.append(detection)
                elif detection['quality']['composite'] >= face_min_quality:
                    # Quality is already good enough (already marked as enhanced=False)
                    final_detections.append(detection)
                # Else: low confidence and/or low quality - skip entirely

            logger.info(f"Enhancement: {len(needs_enhancement)} high-confidence low-quality faces (conf>={face_min_confidence}, quality<{enhancement_quality_trigger}), {len(final_detections)} already good")

            # Enhance frame if needed
            if needs_enhancement:
                enhanced_frame_data = await frame_client.enhance_frame(
                    video_path=video_path,
                    timestamp=timestamp,
                    model=enhancement_model,
                    fidelity_weight=enhancement_fidelity_weight,
                    output_format="jpeg",
                    quality=95
                )

                if enhanced_frame_data:
                    # Decode enhanced frame
                    enhanced_image = cv2.imdecode(
                        np.frombuffer(enhanced_frame_data, np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if enhanced_image is not None:
                        # Re-run detection on enhanced frame
                        enhanced_detections = await self.detect_faces(enhanced_image, face_min_confidence)

                        # Log enhancement results
                        for i, orig in enumerate(needs_enhancement):
                            logger.info(
                                f"Enhancement attempt {i+1}: "
                                f"original_quality={orig['quality']['composite']:.3f}, "
                                f"original_confidence={orig['confidence']:.3f}"
                            )

                        # Match enhanced detections to original (by bbox overlap)
                        for enhanced in enhanced_detections:
                            if enhanced['quality']['composite'] >= face_min_quality:
                                # Mark as enhanced
                                enhanced['enhanced'] = True
                                final_detections.append(enhanced)
                                logger.info(
                                    f"Enhanced face accepted: "
                                    f"quality={enhanced['quality']['composite']:.3f}, "
                                    f"confidence={enhanced['confidence']:.3f}"
                                )
                            else:
                                logger.info(
                                    f"Enhanced face filtered: "
                                    f"quality={enhanced['quality']['composite']:.3f} < {face_min_quality}"
                                )
                    else:
                        logger.error("Failed to decode enhanced frame")
                else:
                    logger.error("Failed to enhance frame")

            logger.info(f"Final: {len(final_detections)} faces after enhancement and filtering")
            return final_detections

        except Exception as e:
            logger.error(f"Error in detect_and_enhance_faces: {e}", exc_info=True)
            # Fallback to original detections with quality filtering
            filtered = [d for d in detections if d['quality']['composite'] >= face_min_quality]
            return filtered

    def _estimate_pose(self, face) -> str:
        """
        Estimate face pose using native InsightFace angles

        Args:
            face: InsightFace face object

        Returns:
            Pose string: front, left, right, front-rotate-left, front-rotate-right
        """
        try:
            # Use native InsightFace pose angles if available
            if hasattr(face, 'pose'):
                pitch, yaw, roll = face.pose  # [pitch, yaw, roll] in degrees

                # Categorize based on yaw (left/right turn) and roll (head tilt)
                if abs(yaw) < 15 and abs(roll) < 15:
                    return "front"
                elif abs(yaw) < 15 and roll < -15:
                    return "front-rotate-left"
                elif abs(yaw) < 15 and roll > 15:
                    return "front-rotate-right"
                elif yaw < -15:
                    return "left"
                else:  # yaw > 15
                    return "right"

            # Fallback to geometric estimation if pose not available
            return self._estimate_pose_geometric(face)

        except Exception as e:
            logger.debug(f"Error estimating pose: {e}")
            return "front"

    def _estimate_pose_geometric(self, face) -> str:
        """
        Fallback: Estimate face pose from keypoint geometry

        Args:
            face: InsightFace face object

        Returns:
            Pose string: front, left, right, front-rotate-left, front-rotate-right
        """
        try:
            # Get landmarks
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

                if abs(eye_to_nose_x) < eye_distance * 0.1:
                    return "front"
                elif eye_to_nose_x > 0:
                    return "right"
                else:
                    return "left"
            elif angle_degrees > 15:
                return "front-rotate-right"
            else:
                return "front-rotate-left"

        except Exception as e:
            logger.debug(f"Error in geometric pose estimation: {e}")
            return "front"


    def cluster_faces(
        self,
        detections: List[Dict],
        similarity_threshold: float = 0.6
    ) -> Dict[str, List[int]]:
        """
        Cluster face detections by embedding similarity

        Args:
            detections: List of face detection dicts
            similarity_threshold: Cosine similarity threshold for same person

        Returns:
            Dict mapping face_id to list of detection indices
        """
        if not detections:
            return {}

        clusters = {}
        next_face_id = 0

        for idx, detection in enumerate(detections):
            embedding = np.array(detection['embedding'])

            # Find matching cluster
            matched = False

            for face_id, detection_indices in clusters.items():
                # Get representative embedding (highest quality detection in cluster)
                rep_idx = detection_indices[0]
                rep_embedding = np.array(detections[rep_idx]['embedding'])

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding, rep_embedding)

                if similarity >= similarity_threshold:
                    clusters[face_id].append(idx)
                    matched = True
                    break

            # Create new cluster if no match
            if not matched:
                face_id = f"face_{next_face_id}"
                clusters[face_id] = [idx]
                next_face_id += 1

        logger.info(f"Clustered {len(detections)} detections into {len(clusters)} unique faces")

        return clusters

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0.0-1.0)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)

    def get_representative_detection(
        self,
        detections: List[Dict],
        indices: List[int]
    ) -> int:
        """
        Get index of best quality detection from cluster

        Args:
            detections: List of all face detections
            indices: Indices of detections in this cluster

        Returns:
            Index of representative detection
        """
        if not indices:
            return -1

        # Find detection with highest quality score
        best_idx = indices[0]
        best_quality = detections[best_idx]['quality']['composite']

        for idx in indices[1:]:
            quality = detections[idx]['quality']['composite']
            if quality > best_quality:
                best_quality = quality
                best_idx = idx

        return best_idx

    async def download_frame(self, url: str, output_path: Path) -> bool:
        """
        Download frame from URL

        Args:
            url: Frame URL
            output_path: Local path to save frame

        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                return True

        except Exception as e:
            logger.error(f"Failed to download frame {url}: {e}")
            return False

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model configuration information

        Returns:
            Dict with model info
        """
        if self.recognition_manager:
            # Using RecognitionManager - return multi-instance info
            return {
                "model_name": self.model_name,
                "device": self.device,
                "det_sizes": list(self.recognition_manager.apps.keys()),
                "num_instances": len(self.recognition_manager.apps),
                "embedding_size": 512,
                "insightface_available": INSIGHTFACE_AVAILABLE,
                "multi_size_enabled": True
            }
        else:
            # Single instance mode
            return {
                "model_name": self.model_name,
                "device": self.device,
                "det_size": self.det_size,
                "embedding_size": 512,
                "insightface_available": INSIGHTFACE_AVAILABLE,
                "multi_size_enabled": False
            }
