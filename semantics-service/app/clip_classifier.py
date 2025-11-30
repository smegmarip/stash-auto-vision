"""
Semantics Service - CLIP/SigLIP Classifier
Zero-shot scene classification and embedding generation
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPClassifier:
    """
    CLIP/SigLIP-based semantic classifier for video frames

    Supports:
    - Zero-shot classification with text prompts
    - Multi-modal embedding generation
    - Batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cuda"
    ):
        """
        Initialize CLIP/SigLIP classifier

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        if device == "cuda" and self.device == "cpu":
            logger.warning("CUDA requested but not available, falling back to CPU")

        self.model = None
        self.processor = None
        self.embedding_dim = 512  # SigLIP-B and CLIP ViT-B/32 both use 512-D

        logger.info(f"Initializing CLIP classifier: {model_name} on {self.device}")

    def load_model(self):
        """Load CLIP/SigLIP model and processor"""
        try:
            # Detect model type and use appropriate classes
            is_siglip = "siglip" in self.model_name.lower()

            if is_siglip:
                from transformers import AutoProcessor, AutoModel
                logger.info(f"Loading SigLIP model: {self.model_name}")
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            else:
                from transformers import CLIPProcessor, CLIPModel
                logger.info(f"Loading CLIP model: {self.model_name}")
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)

            self.model.eval()  # Set to evaluation mode

            # Determine embedding dimension from model config
            if hasattr(self.model.config, 'projection_dim'):
                self.embedding_dim = self.model.config.projection_dim
            elif hasattr(self.model.config, 'vision_config'):
                # For SigLIP and some CLIP variants
                if hasattr(self.model.config.vision_config, 'hidden_size'):
                    self.embedding_dim = self.model.config.vision_config.hidden_size

            model_type = "SigLIP" if is_siglip else "CLIP"
            logger.info(f"{model_type} model loaded successfully (embedding_dim: {self.embedding_dim})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def classify_frames(
        self,
        frames: List[np.ndarray],
        tags: List[str],
        batch_size: int = 32,
        min_confidence: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify frames against predefined tags using zero-shot CLIP

        Args:
            frames: List of frames as numpy arrays (H, W, C)
            tags: List of text prompts/tags for classification
            batch_size: Batch size for inference
            min_confidence: Minimum confidence threshold
            top_k: Maximum tags to return per frame

        Returns:
            List of dicts with frame results
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []

        logger.info(f"Classifying {len(frames)} frames with {len(tags)} tags (batch_size={batch_size})")

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_results = self._classify_batch(batch_frames, tags, min_confidence, top_k)
            results.extend(batch_results)

        logger.info(f"Classification complete: {len(results)} frames processed")
        return results

    def _classify_batch(
        self,
        frames: List[np.ndarray],
        tags: List[str],
        min_confidence: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Classify a single batch of frames

        Args:
            frames: Batch of frames
            tags: Text prompts/tags
            min_confidence: Minimum confidence threshold
            top_k: Maximum tags per frame

        Returns:
            List of classification results
        """
        try:
            # Convert numpy arrays to PIL Images
            pil_images = [Image.fromarray(frame) for frame in frames]

            # Process inputs
            inputs = self.processor(
                text=tags,
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # Shape: (batch_size, num_tags)
                probs = logits_per_image.softmax(dim=1)  # Normalize to probabilities

            # Convert to numpy
            probs_np = probs.cpu().numpy()

            # Build results for each frame in batch
            batch_results = []
            for frame_probs in probs_np:
                # Get top-k tags above confidence threshold
                frame_tags = []
                for tag_idx, confidence in enumerate(frame_probs):
                    if confidence >= min_confidence:
                        frame_tags.append({
                            'tag': tags[tag_idx],
                            'confidence': float(confidence),
                            'source': 'predefined'
                        })

                # Sort by confidence and take top-k
                frame_tags.sort(key=lambda x: x['confidence'], reverse=True)
                frame_tags = frame_tags[:top_k]

                batch_results.append({'tags': frame_tags})

            # Clean up tensors and PIL images
            del outputs
            del logits_per_image
            del probs
            del probs_np
            del inputs
            del pil_images

            return batch_results

        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            # Return empty results for this batch
            return [{'tags': []} for _ in frames]

    def generate_embeddings(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate image embeddings for similarity search

        Args:
            frames: List of frames as numpy arrays
            batch_size: Batch size for inference
            normalize: Whether to L2-normalize embeddings (for cosine similarity)

        Returns:
            Numpy array of embeddings (num_frames, embedding_dim)
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        embeddings = []

        logger.info(f"Generating embeddings for {len(frames)} frames (batch_size={batch_size})")

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_embeddings = self._embed_batch(batch_frames, normalize)
            embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)

        logger.info(f"Embeddings generated: shape={all_embeddings.shape}")
        return all_embeddings

    def _embed_batch(
        self,
        frames: List[np.ndarray],
        normalize: bool
    ) -> np.ndarray:
        """
        Generate embeddings for a single batch

        Args:
            frames: Batch of frames
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        try:
            # Convert numpy arrays to PIL Images
            pil_images = [Image.fromarray(frame) for frame in frames]

            # Process inputs (image only, no text)
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

                if normalize:
                    # L2 normalization for cosine similarity
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embeddings = image_features.cpu().numpy()

            # Clean up tensors and PIL images
            del image_features
            del inputs
            del pil_images

            return embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Return zero embeddings for this batch
            return np.zeros((len(frames), self.embedding_dim), dtype=np.float32)

    def classify_with_custom_prompts(
        self,
        frames: List[np.ndarray],
        prompts: List[str],
        batch_size: int = 32,
        min_confidence: float = 0.5,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify frames using custom natural language prompts

        Args:
            frames: List of frames as numpy arrays
            prompts: List of custom text prompts (e.g., "two people talking in a kitchen")
            batch_size: Batch size for inference
            min_confidence: Minimum confidence threshold
            top_k: Maximum prompts to return per frame

        Returns:
            List of dicts with frame results
        """
        results = self.classify_frames(frames, prompts, batch_size, min_confidence, top_k)

        # Update source to 'custom_prompt'
        for result in results:
            for tag in result['tags']:
                tag['source'] = 'custom_prompt'

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dict with model details
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'loaded': self.model is not None
        }

    def cleanup(self):
        """Free model resources"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model resources cleaned up")
