"""
Captioning Service - JoyCaption Prompt Templates
Prompt type definitions and formatting for JoyCaption VLM
"""

from typing import Dict, Optional
from .models import PromptType


# JoyCaption system prompts for different output styles
PROMPT_TEMPLATES: Dict[PromptType, str] = {
    PromptType.DESCRIPTIVE: """Write a descriptive caption for this image in a formal tone.
Describe the visual elements, composition, and any notable details you observe.
Focus on what is actually visible in the image.""",

    PromptType.DESCRIPTIVE_INFORMAL: """Write a descriptive caption for this image in a casual tone.
Describe what you see as if explaining it to a friend.
Focus on the main elements and overall vibe.""",

    PromptType.STRAIGHTFORWARD: """Write a brief, straightforward caption for this image.
Be direct and concise. Focus only on the most important visual elements.
Use simple language and short sentences.""",

    PromptType.BOORU_LIKE: """Write a list of tags describing this image, similar to image board tags.
Use comma-separated tags in lowercase.
Include tags for: people/characters, actions, setting, objects, clothing, colors, composition.
Order tags by importance, most relevant first.
Use underscores for multi-word tags (e.g., long_hair, red_shirt).
Be specific and comprehensive but avoid redundancy.""",

    PromptType.BOORU_LIKE_EXTENDED: """Write an extended list of booru-style tags for this image.
Use comma-separated tags in lowercase with underscores for spaces.
Be extremely comprehensive and include:
- Number of people and their characteristics
- Actions and poses
- Clothing and accessories
- Setting and background
- Objects and props
- Colors and lighting
- Camera angle and composition
- Mood and atmosphere
Order by relevance. Include both specific and general tags.""",

    PromptType.ART_CRITIC: """Analyze this image as an art critic would.
Discuss composition, use of color, lighting, and visual storytelling.
Consider the mood, atmosphere, and any artistic techniques employed.
Provide a thoughtful, analytical perspective.""",

    PromptType.TRAINING_PROMPT: """Write a detailed prompt that could be used to generate an image like this one.
Include specific details about subjects, composition, lighting, colors, and style.
Format as a single paragraph suitable for image generation models.
Be precise about positions, quantities, and visual characteristics.""",

    PromptType.MLP_TAGS: """Write tags for this image in booru format, focusing on:
- Character attributes (hair color, eye color, expression)
- Actions and poses
- Setting and background elements
- Objects and accessories
Use underscores for multi-word tags. Be specific and comprehensive.""",
}


# Tag categories for organizing output
TAG_CATEGORIES = [
    "person",      # People, characters, demographics
    "action",      # Actions, poses, activities
    "setting",     # Location, environment, background
    "object",      # Props, items, furniture
    "clothing",    # Attire, accessories
    "color",       # Color descriptors
    "composition", # Camera angle, framing
    "mood",        # Atmosphere, emotion
    "style",       # Artistic style descriptors
    "time",        # Time of day, era
    "quantity",    # Number of people/objects
    "body",        # Body parts, features
    "other",       # Uncategorized
]


# Common tag patterns for category detection
CATEGORY_PATTERNS: Dict[str, list] = {
    "person": [
        "man", "woman", "girl", "boy", "person", "people",
        "male", "female", "solo", "couple", "group",
        "1girl", "1boy", "2girls", "2boys", "multiple_girls",
    ],
    "action": [
        "standing", "sitting", "lying", "walking", "running",
        "looking", "holding", "smiling", "talking", "kissing",
        "dancing", "eating", "drinking", "sleeping", "working",
    ],
    "setting": [
        "indoor", "outdoor", "room", "bedroom", "bathroom",
        "kitchen", "office", "street", "park", "beach",
        "forest", "city", "building", "house", "apartment",
    ],
    "object": [
        "chair", "table", "bed", "couch", "sofa",
        "phone", "computer", "book", "bottle", "glass",
        "lamp", "mirror", "window", "door", "car",
    ],
    "clothing": [
        "shirt", "dress", "pants", "skirt", "jacket",
        "hat", "glasses", "jewelry", "shoes", "underwear",
        "bikini", "swimsuit", "uniform", "costume", "nude",
    ],
    "color": [
        "red", "blue", "green", "yellow", "black",
        "white", "pink", "purple", "orange", "brown",
        "blonde", "brunette", "dark", "light", "colorful",
    ],
    "composition": [
        "close_up", "wide_shot", "portrait", "full_body",
        "from_above", "from_below", "from_side", "pov",
        "centered", "profile", "front_view", "back_view",
    ],
    "mood": [
        "happy", "sad", "serious", "playful", "romantic",
        "dark", "bright", "moody", "cheerful", "intense",
        "peaceful", "energetic", "mysterious", "casual",
    ],
    "time": [
        "day", "night", "morning", "evening", "sunset",
        "sunrise", "afternoon", "midnight", "dusk", "dawn",
    ],
    "quantity": [
        "solo", "1", "2", "3", "multiple", "many",
        "single", "pair", "group", "crowd", "alone",
    ],
    "body": [
        "hair", "eyes", "face", "hands", "legs",
        "long_hair", "short_hair", "blue_eyes", "brown_eyes",
    ],
}


def get_prompt_template(prompt_type: PromptType) -> str:
    """Get the prompt template for a given prompt type"""
    return PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES[PromptType.BOORU_LIKE])


def format_prompt(prompt_type: PromptType, custom_prompt: Optional[str] = None) -> str:
    """
    Format the final prompt for JoyCaption

    Args:
        prompt_type: Type of prompt to use
        custom_prompt: Optional custom prompt override

    Returns:
        Formatted prompt string
    """
    if custom_prompt:
        return custom_prompt
    return get_prompt_template(prompt_type)


def detect_tag_category(tag: str) -> str:
    """
    Detect the category of a tag based on patterns

    Args:
        tag: Tag string (may contain underscores)

    Returns:
        Category name
    """
    tag_lower = tag.lower().replace("_", " ").replace("-", " ")
    tag_parts = set(tag_lower.split())

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            pattern_lower = pattern.lower()
            # Check exact match or partial match
            if pattern_lower in tag_lower or pattern_lower in tag_parts:
                return category

    return "other"


def parse_booru_tags(caption: str) -> list:
    """
    Parse booru-style tags from a caption string

    Args:
        caption: Raw caption output from JoyCaption

    Returns:
        List of cleaned tag strings
    """
    # Split on common delimiters
    if "," in caption:
        tags = [t.strip() for t in caption.split(",")]
    elif "\n" in caption:
        tags = [t.strip() for t in caption.split("\n")]
    else:
        tags = [t.strip() for t in caption.split()]

    # Clean tags
    cleaned = []
    for tag in tags:
        tag = tag.strip().lower()
        # Remove common prefixes/suffixes
        tag = tag.strip(".,;:!?()[]{}\"'")
        # Skip empty or very short tags
        if len(tag) >= 2:
            # Convert spaces to underscores (booru style)
            tag = tag.replace(" ", "_")
            # Remove duplicate underscores
            while "__" in tag:
                tag = tag.replace("__", "_")
            cleaned.append(tag)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for tag in cleaned:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique


def estimate_vram_usage(
    batch_size: int = 1,
    use_quantization: bool = True,
    model_variant: str = "alpha-two"
) -> float:
    """
    Estimate VRAM usage for JoyCaption model

    Args:
        batch_size: Number of images to process at once
        use_quantization: Whether using 4-bit quantization
        model_variant: Model variant name

    Returns:
        Estimated VRAM in GB
    """
    # Base model sizes (approximate)
    base_vram = {
        "alpha-two": 17.0,  # Full precision
        "alpha-one": 15.0,
    }

    model_base = base_vram.get(model_variant, 17.0)

    if use_quantization:
        # 4-bit quantization reduces to ~50% of fp16
        model_base = model_base * 0.5

    # Additional VRAM per batch item (approximate)
    per_batch = 0.5

    return model_base + (batch_size - 1) * per_batch
