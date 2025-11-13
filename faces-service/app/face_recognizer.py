"""
Faces Service - Face Recognizer
InsightFace integration for face detection and recognition
"""

import cv2
import numpy as np
import httpx
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

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
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize face recognizer

        Args:
            model_name: InsightFace model (buffalo_l, buffalo_s, buffalo_sc)
            device: cuda or cpu
            det_size: Detection size (width, height)
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not installed")

        self.model_name = model_name
        self.device = device
        self.det_size = det_size

        # Initialize InsightFace
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=det_size)

        logger.info(f"InsightFace initialized: {model_name} on {device}")

    def detect_faces(
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
            # Detect faces
            faces = self.app.get(image)

            logger.debug(f"InsightFace raw detection: found {len(faces)} faces (face_min_confidence={face_min_confidence})")
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

                # Calculate quality score
                quality_score = self._calculate_quality(face, image.shape)

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
                    'quality_score': quality_score,
                    'pose': pose,
                    'demographics': None
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

    def _estimate_pose(self, face) -> str:
        """
        Estimate face pose from landmarks

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
            logger.debug(f"Error estimating pose: {e}")
            return "front"

    def _calculate_quality(self, face, image_shape: Tuple[int, int, int]) -> float:
        """
        Calculate face quality score

        Args:
            face: InsightFace face object
            image_shape: Image shape (height, width, channels)

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            # Start with detection confidence
            quality = face.det_score

            # Factor in face size (larger is better)
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            image_area = image_shape[0] * image_shape[1]
            size_ratio = face_area / image_area

            # Ideal size is 10-30% of image
            if 0.1 <= size_ratio <= 0.3:
                quality *= 1.0
            elif size_ratio < 0.1:
                quality *= (size_ratio / 0.1)  # Penalty for small faces
            else:
                quality *= 0.9  # Slight penalty for very large faces

            # Factor in pose (front-facing is best)
            pose = self._estimate_pose(face)
            if pose == "front":
                quality *= 1.0
            elif "rotate" in pose:
                quality *= 0.9
            else:
                quality *= 0.8

            return min(1.0, quality)

        except Exception as e:
            logger.debug(f"Error calculating quality: {e}")
            return face.det_score

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
        best_quality = detections[best_idx]['quality_score']

        for idx in indices[1:]:
            quality = detections[idx]['quality_score']
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
        return {
            "model_name": self.model_name,
            "device": self.device,
            "det_size": self.det_size,
            "embedding_size": 512,
            "insightface_available": INSIGHTFACE_AVAILABLE
        }
