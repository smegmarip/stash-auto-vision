"""
Captioning Service - Frame Sharpness Detection
Laplacian variance-based sharpness detection for selecting clearest frames
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_laplacian_variance(frame: np.ndarray) -> float:
    """
    Calculate sharpness score using Laplacian variance.

    Higher values indicate sharper (more in-focus) images.

    Args:
        frame: Image as numpy array (H, W, C) in BGR format

    Returns:
        Laplacian variance score (higher = sharper)
    """
    if frame is None or frame.size == 0:
        return 0.0

    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Return variance (measure of edge intensity distribution)
    variance = laplacian.var()

    return float(variance)


def calculate_combined_quality(
    frame: np.ndarray,
    sharpness_weight: float = 0.7,
    contrast_weight: float = 0.2,
    brightness_weight: float = 0.1
) -> Tuple[float, dict]:
    """
    Calculate combined quality score with multiple factors.

    Args:
        frame: Image as numpy array (H, W, C) in BGR format
        sharpness_weight: Weight for Laplacian variance score
        contrast_weight: Weight for contrast score
        brightness_weight: Weight for brightness score (penalizes extremes)

    Returns:
        Tuple of (combined_score, component_scores_dict)
    """
    if frame is None or frame.size == 0:
        return 0.0, {"sharpness": 0.0, "contrast": 0.0, "brightness": 0.0}

    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Sharpness via Laplacian variance (normalize to 0-1 range, cap at 1000)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = min(laplacian.var() / 1000.0, 1.0)

    # Contrast via standard deviation of pixel values
    contrast = min(gray.std() / 128.0, 1.0)  # Normalize (128 is half of 255)

    # Brightness score - penalize too dark or too bright images
    mean_brightness = gray.mean() / 255.0  # Normalize to 0-1
    # Score peaks at 0.5 (ideal brightness), drops toward 0 or 1
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2

    # Calculate combined score
    combined = (
        sharpness * sharpness_weight +
        contrast * contrast_weight +
        brightness_score * brightness_weight
    )

    components = {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness_score,
        "mean_brightness_raw": mean_brightness
    }

    return combined, components


def select_sharpest_frames(
    frames: List[np.ndarray],
    timestamps: List[float],
    top_n: int = 1,
    use_combined_score: bool = False
) -> List[Tuple[int, float, float]]:
    """
    Select the sharpest frames from a list.

    Args:
        frames: List of frames as numpy arrays
        timestamps: Corresponding timestamps for each frame
        top_n: Number of sharpest frames to return
        use_combined_score: Use combined quality score instead of just sharpness

    Returns:
        List of (index, timestamp, score) tuples, sorted by score descending
    """
    if not frames or not timestamps or len(frames) != len(timestamps):
        return []

    scores = []

    for i, frame in enumerate(frames):
        if frame is None:
            scores.append((i, timestamps[i], 0.0))
            continue

        if use_combined_score:
            score, _ = calculate_combined_quality(frame)
        else:
            score = calculate_laplacian_variance(frame)

        scores.append((i, timestamps[i], score))

    # Sort by score descending
    scores.sort(key=lambda x: x[2], reverse=True)

    return scores[:top_n]


def select_sharpest_per_scene(
    frames: List[np.ndarray],
    timestamps: List[float],
    scene_boundaries: List[dict],
    frames_per_scene: int = 1,
    use_combined_score: bool = True,
    min_quality: float = 0.05
) -> List[Tuple[int, float, float, int]]:
    """
    Select the sharpest frames from each scene.

    Args:
        frames: List of all candidate frames
        timestamps: Corresponding timestamps for each frame
        scene_boundaries: Scene boundary information with start/end timestamps
        frames_per_scene: Number of sharpest frames to select per scene
        use_combined_score: Use combined quality score
        min_quality: Minimum quality threshold (0.0-1.0). Frames below this
                     threshold are rejected. Default 0.05 filters black/blank frames.

    Returns:
        List of (frame_index, timestamp, score, scene_index) tuples
    """
    if not frames or not timestamps or not scene_boundaries:
        return []

    results = []

    for scene_idx, scene in enumerate(scene_boundaries):
        start = scene.get("start_timestamp", 0.0)
        end = scene.get("end_timestamp", float('inf'))

        # Find frames within this scene
        scene_frames = []
        for i, ts in enumerate(timestamps):
            if start <= ts <= end and i < len(frames) and frames[i] is not None:
                scene_frames.append((i, ts, frames[i]))

        if not scene_frames:
            logger.debug(f"No frames found for scene {scene_idx} ({start:.2f}s - {end:.2f}s)")
            continue

        # Calculate scores for frames in this scene
        frame_scores = []
        for idx, ts, frame in scene_frames:
            if use_combined_score:
                score, components = calculate_combined_quality(frame)
            else:
                score = calculate_laplacian_variance(frame)
                # Normalize raw Laplacian to 0-1 range for threshold comparison
                score = min(score / 1000.0, 1.0)

            # Skip frames below minimum quality threshold
            if score < min_quality:
                logger.debug(
                    f"Scene {scene_idx}: rejected frame at {ts:.2f}s "
                    f"(score: {score:.4f} < min_quality: {min_quality})"
                )
                continue

            frame_scores.append((idx, ts, score))

        if not frame_scores:
            logger.warning(
                f"Scene {scene_idx} ({start:.2f}s - {end:.2f}s): "
                f"all {len(scene_frames)} candidate frames rejected (below min_quality {min_quality})"
            )
            continue

        # Sort by score and take top N
        frame_scores.sort(key=lambda x: x[2], reverse=True)

        for idx, ts, score in frame_scores[:frames_per_scene]:
            results.append((idx, ts, score, scene_idx))
            logger.debug(f"Scene {scene_idx}: selected frame at {ts:.2f}s (score: {score:.4f})")

    return results


def get_frame_quality_report(frame: np.ndarray) -> dict:
    """
    Generate a detailed quality report for a frame.

    Args:
        frame: Image as numpy array

    Returns:
        Dictionary with quality metrics
    """
    if frame is None or frame.size == 0:
        return {
            "valid": False,
            "reason": "Empty or null frame"
        }

    combined_score, components = calculate_combined_quality(frame)
    laplacian_raw = calculate_laplacian_variance(frame)

    # Classify quality
    if combined_score >= 0.7:
        quality_class = "excellent"
    elif combined_score >= 0.5:
        quality_class = "good"
    elif combined_score >= 0.3:
        quality_class = "fair"
    else:
        quality_class = "poor"

    return {
        "valid": True,
        "combined_score": combined_score,
        "quality_class": quality_class,
        "laplacian_variance": laplacian_raw,
        "sharpness_normalized": components["sharpness"],
        "contrast_normalized": components["contrast"],
        "brightness_score": components["brightness"],
        "mean_brightness": components["mean_brightness_raw"],
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
    }
