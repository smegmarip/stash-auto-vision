"""
Scene summary generator via external OpenAI-compatible LLM API.

Synthesizes frame-level captions and scene metadata into coherent narrative
summaries. Uses the same prompt structure as the training pipeline's
generate_scene_summaries.py to ensure consistency.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are a scene summarizer. Given structured data about a video scene (metadata, participants, and frame-by-frame descriptions), generate a coherent narrative summary.

Guidelines:
- Synthesize frame descriptions into a flowing narrative, not a list
- Describe temporal progression: what happens first, then, throughout, finally
- Focus on actions, interactions, settings, and visual elements
- Ignore mentions of: watermarks, compression artifacts, image quality, "photograph" framing
- Use present tense for describing the scene
- Be specific about: clothing, positions, settings, camera angles, lighting
- Do not speculate about mood, emotions, or intent unless clearly evident
- Length: 2-4 paragraphs covering the entire scene"""

SUMMARY_PROMPT_TEMPLATE = """Summarize this video scene based on the structured data below.

## Scene Metadata
- Duration: {duration_str}
- Frame count: {frame_count} frames sampled
- Resolution: {resolution}

## Participants
{participants_str}

## Promotional Description
{promotional_summary}

## Frame-by-Frame Descriptions
{frame_descriptions}

---

Generate a coherent narrative summary of this scene:"""

# Patterns for cleaning frame captions before summarization
_CAPTION_MEDIUM_RE = re.compile(
    r"^(A|This|The) (photograph|image|picture|photo)\b",
    re.IGNORECASE,
)
_WATERMARK_RE = re.compile(r"[^.]*watermark[^.]*\.", re.IGNORECASE)
_ARTIFACT_RE = re.compile(r"[^.]*compression artifacts?[^.]*\.", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"[^.]*resolution[^.]*\.", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_frame_caption(caption: str) -> str:
    """Clean a frame caption for scene summary context.

    - Replaces "A photograph" / "This image" / etc. with "The frame"
    - Removes sentences mentioning watermarks
    - Removes sentences mentioning compression artifacts
    - Removes sentences mentioning resolution
    - Collapses extra whitespace
    """
    caption = _CAPTION_MEDIUM_RE.sub("The frame", caption)
    caption = _WATERMARK_RE.sub("", caption)
    caption = _ARTIFACT_RE.sub("", caption)
    caption = _RESOLUTION_RE.sub("", caption)
    caption = _WHITESPACE_RE.sub(" ", caption).strip()
    return caption


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds as a human-readable string."""
    if not seconds:
        return "Unknown"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_participants(count: int, genders: Optional[List[str]]) -> str:
    """Format participant info as a descriptive string."""
    if count == 0:
        return "No identified participants"
    gender_str = ", ".join(genders) if genders else "genders unknown"
    return f"{count} participant(s): {gender_str}"


class SummaryGenerator:
    """Generates narrative scene summaries via an external OpenAI-compatible LLM API.

    Designed for the semantics service pipeline. Uses httpx for async HTTP calls
    to a vLLM, llama.cpp, or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 120.0,
    ):
        self.api_base = (
            api_base or os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def _build_prompt(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        """Build the summary prompt from frame captions and metadata.

        Args:
            frame_captions: List of dicts with keys frame_index, timestamp, caption.
            promo_desc: Promotional description of the scene.
            duration: Scene duration in seconds.
            performer_count: Number of performers.
            performer_genders: List of gender strings.
            resolution: Resolution string like "1920x1080".

        Returns:
            Formatted prompt string ready for the LLM.
        """
        duration_str = format_duration(duration if duration else None)
        participants_str = format_participants(performer_count, performer_genders)
        promotional_summary = promo_desc or "Not available"

        # Format frame descriptions with timestamps
        frame_lines: List[str] = []
        for frame in frame_captions:
            timestamp = frame.get("timestamp", 0)
            caption = clean_frame_caption(frame.get("caption", ""))

            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            ts_str = f"{mins:02d}:{secs:02d}"

            frame_lines.append(f"[{ts_str}] {caption}")

        frame_descriptions = "\n".join(frame_lines)

        return SUMMARY_PROMPT_TEMPLATE.format(
            duration_str=duration_str,
            frame_count=len(frame_captions),
            resolution=resolution,
            participants_str=participants_str,
            promotional_summary=promotional_summary,
            frame_descriptions=frame_descriptions,
        )

    async def generate_summary(
        self,
        frame_captions: List[Dict[str, Any]],
        promo_desc: str = "",
        duration: float = 0,
        performer_count: int = 0,
        performer_genders: Optional[List[str]] = None,
        resolution: str = "Unknown",
    ) -> str:
        """Generate a narrative summary from frame captions and metadata.

        Args:
            frame_captions: List of {frame_index, timestamp, caption} dicts.
            promo_desc: Promotional description (optional).
            duration: Scene duration in seconds.
            performer_count: Number of performers.
            performer_genders: List of gender strings.
            resolution: Resolution string like "1920x1080".

        Returns:
            Narrative summary text (2-4 paragraphs).

        Raises:
            httpx.HTTPStatusError: On non-2xx response from LLM API.
            httpx.ConnectError: If LLM API is unreachable.
        """
        prompt = self._build_prompt(
            frame_captions=frame_captions,
            promo_desc=promo_desc,
            duration=duration,
            performer_count=performer_count,
            performer_genders=performer_genders,
            resolution=resolution,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base.rstrip('/')}/chat/completions"

        logger.debug(
            "Requesting summary from %s (model=%s, frames=%d)",
            url,
            self.model,
            len(frame_captions),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()

        try:
            summary = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected API response structure: %s", data)
            raise ValueError("LLM API returned unexpected response format") from exc

        logger.debug("Generated summary (%d chars)", len(summary))
        return summary

    async def is_available(self) -> bool:
        """Check if the LLM API is reachable.

        Returns:
            True if the /models endpoint responds successfully, False otherwise.
        """
        url = f"{self.api_base.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
            return response.status_code == 200
        except httpx.HTTPError:
            logger.debug("LLM API not reachable at %s", url)
            return False
        except Exception:
            logger.debug("Unexpected error checking LLM API availability", exc_info=True)
            return False
