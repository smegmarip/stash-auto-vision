"""
Captioning Service - Tag Aligner
Maps VLM free-form output to fixed Stash tag taxonomy via fuzzy matching
"""

import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import logging

from .models import TagTaxonomyNode, CaptionTag
from .prompt_templates import detect_tag_category

logger = logging.getLogger(__name__)


class TagAligner:
    """Aligns VLM-generated tags to Stash taxonomy"""

    def __init__(
        self,
        taxonomy: List[TagTaxonomyNode],
        similarity_threshold: float = 0.7,
        enable_fuzzy: bool = True
    ):
        """
        Initialize tag aligner

        Args:
            taxonomy: List of Stash tags from taxonomy
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
            enable_fuzzy: Enable fuzzy string matching
        """
        self.taxonomy = taxonomy
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy = enable_fuzzy

        # Build lookup indices
        self._build_indices()

    def _build_indices(self):
        """Build fast lookup indices from taxonomy"""
        # Exact match lookup (lowercase name -> node)
        self.exact_lookup: Dict[str, TagTaxonomyNode] = {}

        # Alias lookup (lowercase alias -> node)
        self.alias_lookup: Dict[str, TagTaxonomyNode] = {}

        # Category lookup (category -> [nodes])
        self.category_lookup: Dict[str, List[TagTaxonomyNode]] = {}

        # All searchable terms for fuzzy matching
        self.all_terms: List[Tuple[str, TagTaxonomyNode]] = []

        for node in self.taxonomy:
            # Exact name match
            name_lower = node.name.lower()
            name_normalized = self._normalize_tag(name_lower)
            self.exact_lookup[name_lower] = node
            self.exact_lookup[name_normalized] = node

            # Add to fuzzy search terms
            self.all_terms.append((name_lower, node))

            # Alias matches
            for alias in node.aliases:
                alias_lower = alias.lower()
                alias_normalized = self._normalize_tag(alias_lower)
                self.alias_lookup[alias_lower] = node
                self.alias_lookup[alias_normalized] = node
                self.all_terms.append((alias_lower, node))

            # Category grouping
            if node.category:
                if node.category not in self.category_lookup:
                    self.category_lookup[node.category] = []
                self.category_lookup[node.category].append(node)

        logger.info(
            f"Tag aligner initialized: {len(self.exact_lookup)} exact matches, "
            f"{len(self.alias_lookup)} aliases, {len(self.all_terms)} fuzzy terms"
        )

    def _normalize_tag(self, tag: str) -> str:
        """
        Normalize tag for matching

        Converts booru-style tags (underscores) to spaces,
        removes special characters, lowercases
        """
        # Replace underscores with spaces
        normalized = tag.replace("_", " ").replace("-", " ")
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized.lower()

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        return SequenceMatcher(None, s1, s2).ratio()

    def _find_best_fuzzy_match(
        self,
        tag: str,
        category_hint: Optional[str] = None
    ) -> Optional[Tuple[TagTaxonomyNode, float]]:
        """
        Find best fuzzy match for a tag

        Args:
            tag: Normalized tag to match
            category_hint: Optional category to prioritize

        Returns:
            (matched node, similarity score) or None
        """
        if not self.enable_fuzzy:
            return None

        best_match = None
        best_score = 0.0

        # If category hint provided, search that category first
        search_terms = self.all_terms
        if category_hint and category_hint in self.category_lookup:
            category_nodes = self.category_lookup[category_hint]
            search_terms = [
                (term, node) for term, node in self.all_terms
                if node in category_nodes
            ]
            # Fall back to all terms if no good match in category
            if not search_terms:
                search_terms = self.all_terms

        for term, node in search_terms:
            score = self._calculate_similarity(tag, term)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = (node, score)

        return best_match

    def align_tag(
        self,
        raw_tag: str,
        confidence: float = 1.0,
        category_hint: Optional[str] = None
    ) -> CaptionTag:
        """
        Align a single tag to taxonomy

        Args:
            raw_tag: Raw tag from VLM output
            confidence: Original confidence score
            category_hint: Optional category hint

        Returns:
            CaptionTag with alignment info
        """
        normalized = self._normalize_tag(raw_tag)
        detected_category = category_hint or detect_tag_category(raw_tag)

        # Try exact match first
        if normalized in self.exact_lookup:
            node = self.exact_lookup[normalized]
            return CaptionTag(
                tag=node.name,
                confidence=confidence,
                source="aligned_exact",
                stash_tag_id=node.id,
                category=node.category or detected_category
            )

        # Try alias match
        if normalized in self.alias_lookup:
            node = self.alias_lookup[normalized]
            return CaptionTag(
                tag=node.name,
                confidence=confidence * 0.95,  # Slight penalty for alias
                source="aligned_alias",
                stash_tag_id=node.id,
                category=node.category or detected_category
            )

        # Try fuzzy match
        fuzzy_result = self._find_best_fuzzy_match(normalized, detected_category)
        if fuzzy_result:
            node, similarity = fuzzy_result
            return CaptionTag(
                tag=node.name,
                confidence=confidence * similarity,  # Scale by similarity
                source="aligned_fuzzy",
                stash_tag_id=node.id,
                category=node.category or detected_category
            )

        # No match - return original tag
        return CaptionTag(
            tag=raw_tag,
            confidence=confidence,
            source="unaligned",
            stash_tag_id=None,
            category=detected_category
        )

    def align_tags(
        self,
        raw_tags: List[str],
        confidences: Optional[List[float]] = None,
        min_confidence: float = 0.5
    ) -> List[CaptionTag]:
        """
        Align multiple tags to taxonomy

        Args:
            raw_tags: List of raw tags from VLM
            confidences: Optional confidence scores for each tag
            min_confidence: Minimum confidence threshold

        Returns:
            List of aligned CaptionTag objects
        """
        if confidences is None:
            # Assume decreasing confidence based on position
            confidences = [
                max(0.5, 1.0 - (i * 0.02))
                for i in range(len(raw_tags))
            ]

        aligned = []
        seen_ids = set()  # Deduplicate by stash_tag_id

        for raw_tag, conf in zip(raw_tags, confidences):
            caption_tag = self.align_tag(raw_tag, conf)

            # Skip low confidence
            if caption_tag.confidence < min_confidence:
                continue

            # Skip duplicates (same Stash tag)
            if caption_tag.stash_tag_id:
                if caption_tag.stash_tag_id in seen_ids:
                    continue
                seen_ids.add(caption_tag.stash_tag_id)

            aligned.append(caption_tag)

        return aligned

    def get_taxonomy_stats(self) -> Dict[str, int]:
        """Get statistics about the loaded taxonomy"""
        return {
            "total_tags": len(self.taxonomy),
            "exact_matches": len(self.exact_lookup),
            "aliases": len(self.alias_lookup),
            "categories": len(self.category_lookup),
            "fuzzy_terms": len(self.all_terms),
        }


def create_default_aligner() -> TagAligner:
    """Create a tag aligner with empty taxonomy (for testing)"""
    return TagAligner(taxonomy=[], enable_fuzzy=True)
