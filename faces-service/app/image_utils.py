"""Image utility functions for faces-service."""

from PIL import Image, ImageOps
import os
import logging

logger = logging.getLogger(__name__)


def normalize_image_if_needed(
    image_path: str,
    output_dir: str = "/tmp/downloads",
    job_id: str = None
) -> tuple[str, bool]:
    """
    Check EXIF orientation and save normalized copy if orientation != 1.

    Args:
        image_path: Path to the source image
        output_dir: Directory to save normalized images
        job_id: Optional job ID for unique filename

    Returns:
        (path_to_use, was_normalized)
        - If normalized: (output_path, True)
        - If not needed: (image_path, False)
    """
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        orientation = exif.get(274)  # 274 = Orientation tag

        if orientation is None or orientation == 1:
            # No rotation needed
            return image_path, False

        # Apply EXIF orientation to pixels
        img_fixed = ImageOps.exif_transpose(img)
        if img_fixed is None:
            return image_path, False

        # Generate output path
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        suffix = f"_{job_id}" if job_id else "_normalized"
        output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")

        # Save without EXIF (pixels are now correctly oriented)
        img_fixed.save(output_path, quality=95)
        logger.info(f"Normalized EXIF orientation {orientation} -> 1: {output_path}")

        return output_path, True

    except Exception as e:
        logger.warning(f"Failed to check/normalize EXIF for {image_path}: {e}")
        return image_path, False


def is_image_file(path: str) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    ext = os.path.splitext(path)[1].lower()
    return ext in image_extensions
