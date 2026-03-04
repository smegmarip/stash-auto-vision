"""
Captioning Service - Hierarchical Tag Scorer
DFS-based hierarchical tag scoring with semantic matching for scene tagging
"""

import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import logging

from .models import TagTaxonomyNode, CaptionTag, FrameCaption, SceneSummaryData

logger = logging.getLogger(__name__)


@dataclass
class TagTreeNode:
    """Node in the hierarchical tag tree"""
    id: str
    name: str
    normalized_name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    normalized_description: Optional[str] = None
    category: Optional[str] = None
    parent_id: Optional[str] = None
    children: List["TagTreeNode"] = field(default_factory=list)
    depth: int = 0

    # Scoring fields populated during traversal
    relevance_score: float = 0.0
    match_type: Optional[str] = None  # exact, alias, semantic, description, hierarchical


@dataclass
class ScoredTag:
    """A tag with computed relevance score"""
    tag_id: str
    tag_name: str
    score: float
    match_type: str
    hierarchy_path: List[str]  # Root to leaf path
    source_evidence: List[str]  # What text matched this tag


class HierarchicalTagger:
    """
    Hierarchical tag scorer using DFS traversal and semantic matching.

    Scores tags against VLM output considering:
    1. Exact/alias string matching
    2. Fuzzy string similarity
    3. Hierarchical inheritance (parent tags boost children)
    4. Semantic similarity (if embeddings available)
    """

    def __init__(
        self,
        taxonomy: List[TagTaxonomyNode],
        hierarchical_decay: float = 0.8,
        min_score_threshold: float = 0.3,
        enable_semantic: bool = False
    ):
        """
        Initialize hierarchical tagger.

        Args:
            taxonomy: Flat list of taxonomy nodes
            hierarchical_decay: Score decay per hierarchy level (0-1)
            min_score_threshold: Minimum score to include tag
            enable_semantic: Enable semantic embedding scoring (requires model)
        """
        self.hierarchical_decay = hierarchical_decay
        self.min_score_threshold = min_score_threshold
        self.enable_semantic = enable_semantic

        # Build tree from flat taxonomy
        self.root_nodes: List[TagTreeNode] = []
        self.node_lookup: Dict[str, TagTreeNode] = {}
        self._build_tree(taxonomy)

        logger.info(
            f"Hierarchical tagger initialized: {len(taxonomy)} tags, "
            f"{len(self.root_nodes)} root nodes"
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        normalized = text.lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _build_tree(self, taxonomy: List[TagTaxonomyNode]):
        """Build tree structure from flat taxonomy list"""
        # Create all nodes first
        for node in taxonomy:
            tree_node = TagTreeNode(
                id=node.id,
                name=node.name,
                normalized_name=self._normalize_text(node.name),
                aliases=[self._normalize_text(a) for a in node.aliases],
                description=node.description,
                normalized_description=self._normalize_text(node.description) if node.description else None,
                category=node.category,
                parent_id=node.parent_id
            )
            self.node_lookup[node.id] = tree_node

        # Link parents to children
        for node_id, tree_node in self.node_lookup.items():
            if tree_node.parent_id and tree_node.parent_id in self.node_lookup:
                parent = self.node_lookup[tree_node.parent_id]
                parent.children.append(tree_node)
                tree_node.depth = parent.depth + 1
            else:
                # Root node
                self.root_nodes.append(tree_node)

        # Sort children for consistent traversal
        for node in self.node_lookup.values():
            node.children.sort(key=lambda x: x.name)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized string similarity"""
        return SequenceMatcher(None, text1, text2).ratio()

    def _score_node_against_text(
        self,
        node: TagTreeNode,
        text_tokens: Set[str],
        full_text: str
    ) -> Tuple[float, str, List[str]]:
        """
        Score a single node against input text.

        Uses tag descriptions for disambiguation when available.
        E.g., "tails" with description "animal appendage" will match
        better when context includes "animal", "fur", etc.

        Returns:
            (score, match_type, evidence_list)
        """
        evidence = []

        # Check exact name match
        name_matched = node.normalized_name in text_tokens or node.normalized_name in full_text

        if name_matched:
            # If we have a description, use it for disambiguation
            if node.normalized_description:
                desc_score = self._score_description_match(
                    node.normalized_description, text_tokens, full_text
                )
                if desc_score > 0.3:
                    # Strong contextual match - boost score
                    return min(1.0, 0.95 + desc_score * 0.1), "exact_with_context", [
                        node.normalized_name,
                        f"context match: {desc_score:.2f}"
                    ]
                elif desc_score < 0.1:
                    # Poor contextual match - this might be wrong tag
                    # Penalize but don't exclude
                    return 0.6, "exact_weak_context", [node.normalized_name]
            # No description or neutral context
            return 0.95 if node.normalized_name in text_tokens else 0.9, "exact", [node.normalized_name]

        # Check aliases
        for alias in node.aliases:
            if alias in text_tokens or alias in full_text:
                # Apply same disambiguation logic for aliases
                if node.normalized_description:
                    desc_score = self._score_description_match(
                        node.normalized_description, text_tokens, full_text
                    )
                    if desc_score > 0.3:
                        return 0.9, "alias_with_context", [alias]
                    elif desc_score < 0.1:
                        return 0.5, "alias_weak_context", [alias]
                return 0.85, "alias", [alias]

        # Check if description terms appear in text (semantic match)
        if node.normalized_description:
            desc_score = self._score_description_match(
                node.normalized_description, text_tokens, full_text
            )
            if desc_score >= 0.5:
                return desc_score * 0.7, "description_match", [
                    f"description terms matched: {desc_score:.2f}"
                ]

        # Fuzzy matching on name
        best_fuzzy = 0.0
        best_fuzzy_token = ""
        for token in text_tokens:
            if len(token) >= 3:  # Skip very short tokens
                sim = self._calculate_text_similarity(node.normalized_name, token)
                if sim > best_fuzzy:
                    best_fuzzy = sim
                    best_fuzzy_token = token

        if best_fuzzy >= 0.75:
            return best_fuzzy * 0.8, "fuzzy", [best_fuzzy_token]

        return 0.0, "none", []

    def _score_description_match(
        self,
        description: str,
        text_tokens: Set[str],
        full_text: str
    ) -> float:
        """
        Score how well text matches a tag's description.

        Used for disambiguation - e.g., for tag "tails" with description
        "animal appendage, not coin flip", check if context supports
        the animal interpretation.

        Returns:
            Score 0.0-1.0 indicating contextual match strength
        """
        if not description:
            return 0.5  # Neutral - no description to check

        desc_tokens = set(description.split())

        # Remove common stop words
        stop_words = {"a", "an", "the", "is", "are", "not", "or", "and", "of", "for", "to"}
        desc_tokens = {t for t in desc_tokens if t not in stop_words and len(t) > 2}

        if not desc_tokens:
            return 0.5

        # Count how many description terms appear in input
        matches = 0
        for token in desc_tokens:
            if token in text_tokens or token in full_text:
                matches += 1

        # Also check for negative indicators (words after "not")
        # E.g., "not coin flip" - if "coin" or "flip" appears, lower score
        # This is a simple heuristic

        return matches / len(desc_tokens)

    def _dfs_score(
        self,
        node: TagTreeNode,
        text_tokens: Set[str],
        full_text: str,
        parent_score: float = 0.0,
        hierarchy_path: List[str] = None
    ) -> List[ScoredTag]:
        """
        DFS pre-order traversal scoring.

        Visits nodes in pre-order (parent before children).
        Children inherit boosted scores from matching parents.
        """
        if hierarchy_path is None:
            hierarchy_path = []

        current_path = hierarchy_path + [node.name]
        results = []

        # Score this node
        direct_score, match_type, evidence = self._score_node_against_text(
            node, text_tokens, full_text
        )

        # Combine direct score with inherited parent score
        inherited_score = parent_score * self.hierarchical_decay
        final_score = max(direct_score, inherited_score)

        # Determine effective match type
        if direct_score > 0:
            effective_match_type = match_type
        elif inherited_score > 0:
            effective_match_type = "hierarchical"
            evidence = [f"inherited from {hierarchy_path[-1]}"] if hierarchy_path else []
        else:
            effective_match_type = "none"

        # Add to results if score exceeds threshold
        if final_score >= self.min_score_threshold:
            results.append(ScoredTag(
                tag_id=node.id,
                tag_name=node.name,
                score=final_score,
                match_type=effective_match_type,
                hierarchy_path=current_path,
                source_evidence=evidence
            ))

        # Recursively process children (pre-order DFS)
        child_parent_score = direct_score if direct_score > inherited_score else inherited_score

        for child in node.children:
            child_results = self._dfs_score(
                child, text_tokens, full_text,
                parent_score=child_parent_score,
                hierarchy_path=current_path
            )
            results.extend(child_results)

        return results

    def score_text(self, text: str) -> List[ScoredTag]:
        """
        Score all tags against input text using DFS traversal.

        Args:
            text: Input text from VLM (caption, description, etc.)

        Returns:
            List of ScoredTag objects, sorted by score descending
        """
        if not text:
            return []

        # Normalize and tokenize
        full_text = self._normalize_text(text)
        text_tokens = set(full_text.split())

        # Also add multi-word phrases (up to 3 words)
        words = full_text.split()
        for i in range(len(words)):
            for j in range(i + 2, min(i + 4, len(words) + 1)):
                phrase = " ".join(words[i:j])
                text_tokens.add(phrase)

        # DFS traverse all root nodes
        all_results = []
        for root in self.root_nodes:
            results = self._dfs_score(root, text_tokens, full_text)
            all_results.extend(results)

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results

    def score_frame_caption(self, frame: FrameCaption) -> List[ScoredTag]:
        """
        Score tags for a single frame caption.

        Combines raw caption and summary data for comprehensive scoring.
        """
        texts_to_score = [frame.raw_caption]

        # Add structured summary data if available
        if frame.summary:
            summary = frame.summary
            if summary.setting:
                texts_to_score.append(summary.setting)
            if summary.locale:
                texts_to_score.append(summary.locale)
            if summary.mood:
                texts_to_score.append(summary.mood)
            if summary.genre:
                texts_to_score.append(summary.genre)
            texts_to_score.extend(summary.objects)
            texts_to_score.extend(summary.activities)
            texts_to_score.extend(summary.attire)
            if summary.narrative_context:
                texts_to_score.append(summary.narrative_context)

        # Combine all texts
        combined_text = " ".join(texts_to_score)
        return self.score_text(combined_text)

    def aggregate_scene_tags(
        self,
        frames: List[FrameCaption],
        top_n: int = 20,
        weight_by_sharpness: bool = True
    ) -> List[ScoredTag]:
        """
        Aggregate frame tags into scene-level tags.

        Args:
            frames: List of frame captions in a scene
            top_n: Maximum tags to return
            weight_by_sharpness: Weight frame contributions by sharpness score

        Returns:
            Aggregated and deduplicated scene tags
        """
        if not frames:
            return []

        # Score each frame
        all_scored: Dict[str, List[Tuple[float, ScoredTag]]] = {}

        for frame in frames:
            frame_results = self.score_frame_caption(frame)

            # Determine frame weight
            weight = 1.0
            if weight_by_sharpness:
                # Could get from frame metadata if available
                weight = 1.0

            for scored in frame_results:
                if scored.tag_id not in all_scored:
                    all_scored[scored.tag_id] = []
                all_scored[scored.tag_id].append((weight, scored))

        # Aggregate scores per tag
        aggregated = []
        for tag_id, weighted_scores in all_scored.items():
            # Weighted average score
            total_weight = sum(w for w, _ in weighted_scores)
            avg_score = sum(w * s.score for w, s in weighted_scores) / total_weight

            # Boost for appearing in multiple frames
            occurrence_boost = min(1.2, 1.0 + (len(weighted_scores) - 1) * 0.05)
            final_score = avg_score * occurrence_boost

            # Use first occurrence for metadata
            _, first = weighted_scores[0]
            aggregated.append(ScoredTag(
                tag_id=tag_id,
                tag_name=first.tag_name,
                score=final_score,
                match_type=first.match_type,
                hierarchy_path=first.hierarchy_path,
                source_evidence=first.source_evidence
            ))

        # Sort and limit
        aggregated.sort(key=lambda x: x.score, reverse=True)
        return aggregated[:top_n]

    def convert_to_caption_tags(
        self,
        scored_tags: List[ScoredTag],
        source: str = "hierarchical"
    ) -> List[CaptionTag]:
        """Convert ScoredTag objects to CaptionTag model objects"""
        return [
            CaptionTag(
                tag=st.tag_name,
                confidence=st.score,
                source=source,
                stash_tag_id=st.tag_id,
                category=None  # Could extract from hierarchy path
            )
            for st in scored_tags
        ]

    def get_hierarchy_stats(self) -> Dict:
        """Get statistics about the tag hierarchy"""
        max_depth = 0
        total_with_children = 0

        def traverse(node: TagTreeNode, depth: int):
            nonlocal max_depth, total_with_children
            max_depth = max(max_depth, depth)
            if node.children:
                total_with_children += 1
            for child in node.children:
                traverse(child, depth + 1)

        for root in self.root_nodes:
            traverse(root, 0)

        return {
            "total_tags": len(self.node_lookup),
            "root_nodes": len(self.root_nodes),
            "max_depth": max_depth,
            "tags_with_children": total_with_children
        }
