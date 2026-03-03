#!/usr/bin/env python3
"""
Taxonomy Seed Script for Stash

Creates a hierarchical tag taxonomy in Stash for testing captioner tag alignment.
Covers social media clip semantics: locale, subject, activity, context, etc.

Usage:
    python seed_taxonomy.py --stash-url http://localhost:9999 --api-key YOUR_KEY

    # Dry run (show what would be created):
    python seed_taxonomy.py --dry-run
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import httpx


@dataclass
class TagDefinition:
    """Definition of a tag to create in Stash"""
    name: str
    description: str
    aliases: List[str] = field(default_factory=list)
    children: List['TagDefinition'] = field(default_factory=list)


# =============================================================================
# COMPREHENSIVE TAG TAXONOMY
# Hierarchical structure covering social media clip semantics
# =============================================================================

TAXONOMY: List[TagDefinition] = [
    # =========================================================================
    # SUBJECT - Who/what is in the frame
    # =========================================================================
    TagDefinition(
        name="Subject",
        description="Primary subjects visible in the frame - people, characters, or focal points",
        children=[
            TagDefinition(
                name="Person Count",
                description="Number of people visible in the frame",
                children=[
                    TagDefinition(
                        name="Solo",
                        description="Single person alone in the frame with no other people visible",
                        aliases=["1girl", "1boy", "1person", "single", "alone"]
                    ),
                    TagDefinition(
                        name="Duo",
                        description="Exactly two people visible in the frame",
                        aliases=["2girls", "2boys", "couple", "pair", "two_people"]
                    ),
                    TagDefinition(
                        name="Group",
                        description="Three or more people visible in the frame",
                        aliases=["multiple_people", "crowd", "gathering", "3+"]
                    ),
                ],
            ),
            TagDefinition(
                name="Gender Presentation",
                description="Apparent gender presentation of subjects",
                children=[
                    TagDefinition(
                        name="Female Presenting",
                        description="Subject presents with traditionally feminine appearance",
                        aliases=["woman", "girl", "female", "feminine"]
                    ),
                    TagDefinition(
                        name="Male Presenting",
                        description="Subject presents with traditionally masculine appearance",
                        aliases=["man", "boy", "male", "masculine"]
                    ),
                ],
            ),
            TagDefinition(
                name="Age Appearance",
                description="Apparent age range of subjects",
                children=[
                    TagDefinition(
                        name="Young Adult",
                        description="Subject appears to be in their late teens to early thirties",
                        aliases=["20s", "young", "youthful"]
                    ),
                    TagDefinition(
                        name="Middle Aged",
                        description="Subject appears to be in their forties to fifties",
                        aliases=["mature", "40s", "50s"]
                    ),
                    TagDefinition(
                        name="Elderly",
                        description="Subject appears to be senior aged",
                        aliases=["old", "senior", "aged"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # PHYSICAL ATTRIBUTES - Body and appearance characteristics
    # =========================================================================
    TagDefinition(
        name="Physical Attributes",
        description="Observable physical characteristics of subjects",
        children=[
            TagDefinition(
                name="Hair",
                description="Hair characteristics and styling",
                children=[
                    TagDefinition(
                        name="Hair Length",
                        description="Length of hair",
                        children=[
                            TagDefinition(
                                name="Long Hair",
                                description="Hair extending past the shoulders",
                                aliases=["long_hair", "very_long_hair"]
                            ),
                            TagDefinition(
                                name="Medium Hair",
                                description="Hair at shoulder length",
                                aliases=["medium_hair", "shoulder_length"]
                            ),
                            TagDefinition(
                                name="Short Hair",
                                description="Hair above the shoulders",
                                aliases=["short_hair", "pixie_cut"]
                            ),
                            TagDefinition(
                                name="Bald",
                                description="No hair or shaved head",
                                aliases=["bald", "shaved_head", "hairless"]
                            ),
                        ],
                    ),
                    TagDefinition(
                        name="Hair Color",
                        description="Color of hair",
                        children=[
                            TagDefinition(
                                name="Blonde Hair",
                                description="Light yellow to golden colored hair",
                                aliases=["blonde", "blond", "golden_hair", "yellow_hair"]
                            ),
                            TagDefinition(
                                name="Brown Hair",
                                description="Brown colored hair in various shades",
                                aliases=["brunette", "brown_hair", "chestnut_hair"]
                            ),
                            TagDefinition(
                                name="Black Hair",
                                description="Very dark brown to black colored hair",
                                aliases=["black_hair", "dark_hair", "raven_hair"]
                            ),
                            TagDefinition(
                                name="Red Hair",
                                description="Red, auburn, or ginger colored hair",
                                aliases=["redhead", "red_hair", "ginger", "auburn"]
                            ),
                            TagDefinition(
                                name="Gray Hair",
                                description="Gray or white colored hair",
                                aliases=["grey_hair", "white_hair", "silver_hair"]
                            ),
                            TagDefinition(
                                name="Dyed Hair",
                                description="Artificially colored hair in unnatural shades",
                                aliases=["colored_hair", "pink_hair", "blue_hair", "purple_hair", "green_hair"]
                            ),
                        ],
                    ),
                ],
            ),
            TagDefinition(
                name="Eye Color",
                description="Color of eyes",
                children=[
                    TagDefinition(
                        name="Blue Eyes",
                        description="Blue colored irises",
                        aliases=["blue_eyes"]
                    ),
                    TagDefinition(
                        name="Brown Eyes",
                        description="Brown colored irises",
                        aliases=["brown_eyes"]
                    ),
                    TagDefinition(
                        name="Green Eyes",
                        description="Green colored irises",
                        aliases=["green_eyes"]
                    ),
                ],
            ),
            TagDefinition(
                name="Body Type",
                description="General body build and physique",
                children=[
                    TagDefinition(
                        name="Slim",
                        description="Slender or thin body type",
                        aliases=["thin", "slender", "skinny", "petite"]
                    ),
                    TagDefinition(
                        name="Athletic",
                        description="Toned, muscular, or fit body type",
                        aliases=["fit", "toned", "muscular", "athletic_build"]
                    ),
                    TagDefinition(
                        name="Curvy",
                        description="Body type with pronounced curves",
                        aliases=["curvy", "voluptuous", "full_figured"]
                    ),
                    TagDefinition(
                        name="Plus Size",
                        description="Larger body type",
                        aliases=["plus_size", "thick", "bbw"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # APPAREL - Clothing and accessories
    # =========================================================================
    TagDefinition(
        name="Apparel",
        description="Clothing, garments, and worn accessories visible on subjects",
        children=[
            TagDefinition(
                name="Upper Body Clothing",
                description="Garments worn on the upper body",
                children=[
                    TagDefinition(
                        name="Shirt",
                        description="General upper body garment with sleeves",
                        aliases=["shirt", "top", "blouse"]
                    ),
                    TagDefinition(
                        name="T-Shirt",
                        description="Casual short-sleeved shirt typically made of cotton",
                        aliases=["t-shirt", "tee", "tshirt"]
                    ),
                    TagDefinition(
                        name="Tank Top",
                        description="Sleeveless upper body garment",
                        aliases=["tank_top", "singlet", "sleeveless"]
                    ),
                    TagDefinition(
                        name="Sweater",
                        description="Knitted upper body garment for warmth",
                        aliases=["sweater", "jumper", "pullover", "knit"]
                    ),
                    TagDefinition(
                        name="Jacket",
                        description="Outerwear garment worn over other clothing",
                        aliases=["jacket", "coat", "blazer", "outerwear"]
                    ),
                    TagDefinition(
                        name="Hoodie",
                        description="Hooded sweatshirt",
                        aliases=["hoodie", "hooded_sweatshirt", "hooded_jacket"]
                    ),
                ],
            ),
            TagDefinition(
                name="Lower Body Clothing",
                description="Garments worn on the lower body",
                children=[
                    TagDefinition(
                        name="Pants",
                        description="Full-length leg covering garment",
                        aliases=["pants", "trousers", "slacks"]
                    ),
                    TagDefinition(
                        name="Jeans",
                        description="Denim pants",
                        aliases=["jeans", "denim", "denim_pants"]
                    ),
                    TagDefinition(
                        name="Shorts",
                        description="Short pants ending above the knee",
                        aliases=["shorts", "short_pants"]
                    ),
                    TagDefinition(
                        name="Skirt",
                        description="Garment hanging from the waist covering lower body",
                        aliases=["skirt", "miniskirt", "long_skirt"]
                    ),
                    TagDefinition(
                        name="Leggings",
                        description="Tight-fitting leg garment",
                        aliases=["leggings", "yoga_pants", "tights"]
                    ),
                ],
            ),
            TagDefinition(
                name="Full Body Clothing",
                description="Garments covering the entire body",
                children=[
                    TagDefinition(
                        name="Dress",
                        description="One-piece garment covering torso and extending down",
                        aliases=["dress", "gown", "frock"]
                    ),
                    TagDefinition(
                        name="Jumpsuit",
                        description="One-piece garment with legs",
                        aliases=["jumpsuit", "romper", "overalls"]
                    ),
                    TagDefinition(
                        name="Bodysuit",
                        description="Tight-fitting one-piece garment",
                        aliases=["bodysuit", "leotard", "catsuit"]
                    ),
                ],
            ),
            TagDefinition(
                name="Swimwear",
                description="Clothing designed for swimming or beach activities",
                children=[
                    TagDefinition(
                        name="Bikini",
                        description="Two-piece swimsuit",
                        aliases=["bikini", "two_piece"]
                    ),
                    TagDefinition(
                        name="One-Piece Swimsuit",
                        description="Single piece swimming garment",
                        aliases=["one-piece", "swimsuit", "bathing_suit"]
                    ),
                    TagDefinition(
                        name="Swim Trunks",
                        description="Shorts designed for swimming",
                        aliases=["swim_trunks", "board_shorts", "swimming_shorts"]
                    ),
                ],
            ),
            TagDefinition(
                name="Underwear",
                description="Undergarments worn beneath outer clothing",
                children=[
                    TagDefinition(
                        name="Bra",
                        description="Undergarment supporting the breasts",
                        aliases=["bra", "brassiere", "sports_bra"]
                    ),
                    TagDefinition(
                        name="Panties",
                        description="Underwear covering the pelvic region",
                        aliases=["panties", "underwear", "briefs", "thong"]
                    ),
                    TagDefinition(
                        name="Lingerie",
                        description="Decorative intimate apparel",
                        aliases=["lingerie", "intimate_apparel", "negligee"]
                    ),
                ],
            ),
            TagDefinition(
                name="Uniform",
                description="Clothing of distinctive design worn by members of a particular group serving as identification",
                children=[
                    TagDefinition(
                        name="School Uniform",
                        description="Standardized clothing worn by students at educational institutions",
                        aliases=["school_uniform", "student_uniform"]
                    ),
                    TagDefinition(
                        name="Work Uniform",
                        description="Standardized clothing worn for occupational identification",
                        aliases=["work_uniform", "occupational_uniform", "scrubs"]
                    ),
                    TagDefinition(
                        name="Military Uniform",
                        description="Standardized clothing worn by armed forces members",
                        aliases=["military_uniform", "army_uniform", "camo"]
                    ),
                    TagDefinition(
                        name="Sports Uniform",
                        description="Standardized athletic clothing for team identification",
                        aliases=["sports_uniform", "jersey", "team_uniform"]
                    ),
                ],
            ),
            TagDefinition(
                name="Costume",
                description="Special clothing worn for performance, celebration, or roleplay",
                aliases=["costume", "cosplay", "outfit"],
            ),
            TagDefinition(
                name="Nude",
                description="Subject wearing no clothing",
                aliases=["nude", "naked", "unclothed", "no_clothes"]
            ),
            TagDefinition(
                name="Accessories",
                description="Items worn in addition to main clothing",
                children=[
                    TagDefinition(
                        name="Hat",
                        description="Head covering worn for style or protection",
                        aliases=["hat", "cap", "beanie", "headwear"]
                    ),
                    TagDefinition(
                        name="Glasses",
                        description="Eyewear for vision correction or style",
                        aliases=["glasses", "eyeglasses", "spectacles", "sunglasses"]
                    ),
                    TagDefinition(
                        name="Jewelry",
                        description="Decorative items worn on the body",
                        aliases=["jewelry", "necklace", "earrings", "bracelet", "ring"]
                    ),
                    TagDefinition(
                        name="Watch",
                        description="Timepiece worn on the wrist",
                        aliases=["watch", "wristwatch"]
                    ),
                    TagDefinition(
                        name="Bag",
                        description="Carried container for personal items",
                        aliases=["bag", "purse", "handbag", "backpack"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # SETTING - Location and environment
    # =========================================================================
    TagDefinition(
        name="Setting",
        description="The location, environment, or backdrop where the scene takes place",
        children=[
            TagDefinition(
                name="Indoor",
                description="Scene takes place inside a building or enclosed structure",
                aliases=["indoor", "indoors", "inside", "interior"],
                children=[
                    TagDefinition(
                        name="Bedroom",
                        description="Private room designed for sleeping with a bed as the central feature",
                        aliases=["bedroom", "bed_room", "sleeping_room"]
                    ),
                    TagDefinition(
                        name="Bathroom",
                        description="Room containing facilities for bathing and personal hygiene",
                        aliases=["bathroom", "bath", "restroom", "shower_room"]
                    ),
                    TagDefinition(
                        name="Living Room",
                        description="Common room for relaxation and socializing in a residence",
                        aliases=["living_room", "lounge", "sitting_room", "den"]
                    ),
                    TagDefinition(
                        name="Kitchen",
                        description="Room equipped for food preparation and cooking",
                        aliases=["kitchen", "cooking_area"]
                    ),
                    TagDefinition(
                        name="Office",
                        description="Room designed for work or professional activities",
                        aliases=["office", "workspace", "study", "home_office"]
                    ),
                    TagDefinition(
                        name="Gym",
                        description="Facility equipped for physical exercise and training",
                        aliases=["gym", "fitness_center", "workout_room", "exercise_room"]
                    ),
                    TagDefinition(
                        name="Studio",
                        description="Room designed for creative work, photography, or recording",
                        aliases=["studio", "photo_studio", "recording_studio"]
                    ),
                    TagDefinition(
                        name="Hotel Room",
                        description="Temporary accommodation in a commercial lodging establishment",
                        aliases=["hotel_room", "hotel", "motel_room"]
                    ),
                ],
            ),
            TagDefinition(
                name="Outdoor",
                description="Scene takes place outside in an open environment",
                aliases=["outdoor", "outdoors", "outside", "exterior"],
                children=[
                    TagDefinition(
                        name="Beach",
                        description="Sandy or pebbly shore beside a body of water",
                        aliases=["beach", "shore", "seaside", "coast"]
                    ),
                    TagDefinition(
                        name="Pool",
                        description="Artificial body of water for swimming",
                        aliases=["pool", "swimming_pool", "poolside"]
                    ),
                    TagDefinition(
                        name="Park",
                        description="Public green space with trees and recreational areas",
                        aliases=["park", "garden", "public_garden"]
                    ),
                    TagDefinition(
                        name="Street",
                        description="Public road in an urban or suburban area",
                        aliases=["street", "road", "sidewalk", "urban"]
                    ),
                    TagDefinition(
                        name="Forest",
                        description="Large area covered with trees and undergrowth",
                        aliases=["forest", "woods", "woodland", "trees"]
                    ),
                    TagDefinition(
                        name="Mountain",
                        description="Elevated natural landform rising above surroundings",
                        aliases=["mountain", "mountains", "hills", "hiking"]
                    ),
                    TagDefinition(
                        name="Rooftop",
                        description="Top surface of a building used as an outdoor space",
                        aliases=["rooftop", "roof", "terrace"]
                    ),
                    TagDefinition(
                        name="Balcony",
                        description="Platform projecting from a building with a railing",
                        aliases=["balcony", "patio", "deck"]
                    ),
                ],
            ),
            TagDefinition(
                name="Vehicle",
                description="Scene takes place in or around a mode of transportation",
                children=[
                    TagDefinition(
                        name="Car",
                        description="Inside or around a personal automobile",
                        aliases=["car", "automobile", "vehicle", "car_interior"]
                    ),
                    TagDefinition(
                        name="Boat",
                        description="Watercraft or vessel",
                        aliases=["boat", "yacht", "ship"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # ACTIVITY - What is happening
    # =========================================================================
    TagDefinition(
        name="Activity",
        description="Actions, behaviors, or activities being performed by subjects",
        children=[
            TagDefinition(
                name="Pose",
                description="Static body positions and stances",
                children=[
                    TagDefinition(
                        name="Standing",
                        description="Upright position on feet",
                        aliases=["standing", "stand", "upright"]
                    ),
                    TagDefinition(
                        name="Sitting",
                        description="Seated position on a surface",
                        aliases=["sitting", "seated", "sit"]
                    ),
                    TagDefinition(
                        name="Lying Down",
                        description="Horizontal position on a surface",
                        aliases=["lying", "lying_down", "reclining", "prone", "supine"]
                    ),
                    TagDefinition(
                        name="Kneeling",
                        description="Position with one or both knees on the ground",
                        aliases=["kneeling", "kneel", "on_knees"]
                    ),
                    TagDefinition(
                        name="Bending",
                        description="Body curved or folded forward",
                        aliases=["bending", "bent_over", "leaning"]
                    ),
                    TagDefinition(
                        name="Squatting",
                        description="Crouching position with bent knees",
                        aliases=["squatting", "squat", "crouching"]
                    ),
                ],
            ),
            TagDefinition(
                name="Movement",
                description="Dynamic actions involving motion",
                children=[
                    TagDefinition(
                        name="Walking",
                        description="Moving on foot at a regular pace",
                        aliases=["walking", "walk", "strolling"]
                    ),
                    TagDefinition(
                        name="Running",
                        description="Moving rapidly on foot",
                        aliases=["running", "run", "jogging", "sprinting"]
                    ),
                    TagDefinition(
                        name="Dancing",
                        description="Moving rhythmically to music",
                        aliases=["dancing", "dance", "twerking"]
                    ),
                    TagDefinition(
                        name="Jumping",
                        description="Pushing off the ground into the air",
                        aliases=["jumping", "jump", "leaping"]
                    ),
                    TagDefinition(
                        name="Swimming",
                        description="Moving through water using the body",
                        aliases=["swimming", "swim", "in_water"]
                    ),
                ],
            ),
            TagDefinition(
                name="Expression",
                description="Facial expressions and emotional displays",
                children=[
                    TagDefinition(
                        name="Smiling",
                        description="Facial expression with upturned corners of mouth",
                        aliases=["smiling", "smile", "grinning", "happy"]
                    ),
                    TagDefinition(
                        name="Laughing",
                        description="Expressing amusement with vocal sounds",
                        aliases=["laughing", "laugh", "giggling"]
                    ),
                    TagDefinition(
                        name="Serious",
                        description="Neutral or stern facial expression",
                        aliases=["serious", "neutral", "stoic"]
                    ),
                    TagDefinition(
                        name="Pouting",
                        description="Pushing lips forward in expression",
                        aliases=["pouting", "pout", "duck_face"]
                    ),
                    TagDefinition(
                        name="Winking",
                        description="Briefly closing one eye",
                        aliases=["winking", "wink", "one_eye_closed"]
                    ),
                ],
            ),
            TagDefinition(
                name="Looking Direction",
                description="Where the subject's gaze is directed",
                children=[
                    TagDefinition(
                        name="Looking at Camera",
                        description="Subject's gaze directed toward the viewer",
                        aliases=["looking_at_viewer", "looking_at_camera", "eye_contact"]
                    ),
                    TagDefinition(
                        name="Looking Away",
                        description="Subject's gaze directed away from camera",
                        aliases=["looking_away", "averted_gaze", "looking_to_side"]
                    ),
                    TagDefinition(
                        name="Looking Down",
                        description="Subject's gaze directed downward",
                        aliases=["looking_down", "downcast_eyes"]
                    ),
                ],
            ),
            TagDefinition(
                name="Interaction",
                description="Activities involving multiple people or objects",
                children=[
                    TagDefinition(
                        name="Talking",
                        description="Verbal communication between subjects",
                        aliases=["talking", "conversation", "speaking", "chatting"]
                    ),
                    TagDefinition(
                        name="Kissing",
                        description="Pressing lips against another person or object",
                        aliases=["kissing", "kiss"]
                    ),
                    TagDefinition(
                        name="Hugging",
                        description="Embracing another person with arms",
                        aliases=["hugging", "hug", "embrace", "embracing"]
                    ),
                    TagDefinition(
                        name="Holding Hands",
                        description="Grasping hands with another person",
                        aliases=["holding_hands", "hand_holding"]
                    ),
                ],
            ),
            TagDefinition(
                name="Self-Care",
                description="Personal grooming and care activities",
                children=[
                    TagDefinition(
                        name="Applying Makeup",
                        description="Putting on cosmetic products",
                        aliases=["makeup", "applying_makeup", "cosmetics"]
                    ),
                    TagDefinition(
                        name="Showering",
                        description="Bathing under a spray of water",
                        aliases=["showering", "shower", "bathing"]
                    ),
                    TagDefinition(
                        name="Stretching",
                        description="Extending the body or limbs",
                        aliases=["stretching", "stretch", "yoga"]
                    ),
                ],
            ),
            TagDefinition(
                name="Using Device",
                description="Interacting with electronic devices",
                children=[
                    TagDefinition(
                        name="Using Phone",
                        description="Operating a mobile phone",
                        aliases=["phone", "smartphone", "selfie", "texting"]
                    ),
                    TagDefinition(
                        name="Taking Photo",
                        description="Capturing an image with a camera",
                        aliases=["photography", "taking_photo", "camera"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # COMPOSITION - How the shot is framed
    # =========================================================================
    TagDefinition(
        name="Composition",
        description="Technical aspects of how the image is framed and captured",
        children=[
            TagDefinition(
                name="Shot Type",
                description="How much of the subject is visible in frame",
                children=[
                    TagDefinition(
                        name="Close-Up",
                        description="Tight framing on face or specific detail",
                        aliases=["close_up", "closeup", "face_focus"]
                    ),
                    TagDefinition(
                        name="Medium Shot",
                        description="Subject framed from waist up",
                        aliases=["medium_shot", "waist_up", "upper_body"]
                    ),
                    TagDefinition(
                        name="Full Body",
                        description="Entire body visible in frame",
                        aliases=["full_body", "full_shot", "whole_body"]
                    ),
                    TagDefinition(
                        name="Wide Shot",
                        description="Subject shown with significant surroundings",
                        aliases=["wide_shot", "establishing_shot", "long_shot"]
                    ),
                ],
            ),
            TagDefinition(
                name="Camera Angle",
                description="Vertical angle from which the image is captured",
                children=[
                    TagDefinition(
                        name="Eye Level",
                        description="Camera at subject's eye height",
                        aliases=["eye_level", "straight_on"]
                    ),
                    TagDefinition(
                        name="High Angle",
                        description="Camera positioned above subject looking down",
                        aliases=["high_angle", "from_above", "looking_down_at"]
                    ),
                    TagDefinition(
                        name="Low Angle",
                        description="Camera positioned below subject looking up",
                        aliases=["low_angle", "from_below", "looking_up_at", "worms_eye"]
                    ),
                    TagDefinition(
                        name="Dutch Angle",
                        description="Camera tilted on its horizontal axis",
                        aliases=["dutch_angle", "tilted", "canted"]
                    ),
                ],
            ),
            TagDefinition(
                name="POV",
                description="Point-of-view perspective",
                children=[
                    TagDefinition(
                        name="First Person POV",
                        description="Shot from the viewpoint of a participant",
                        aliases=["pov", "first_person", "subjective_shot"]
                    ),
                    TagDefinition(
                        name="Selfie",
                        description="Self-portrait typically with phone at arm's length",
                        aliases=["selfie", "self_shot", "mirror_selfie"]
                    ),
                    TagDefinition(
                        name="Mirror Shot",
                        description="Subject photographed via reflection",
                        aliases=["mirror", "mirror_shot", "reflection"]
                    ),
                ],
            ),
            TagDefinition(
                name="Focus",
                description="What part of the frame is in sharp focus",
                children=[
                    TagDefinition(
                        name="Face Focus",
                        description="Face is the sharp focal point",
                        aliases=["face_focus", "portrait"]
                    ),
                    TagDefinition(
                        name="Body Focus",
                        description="Body is the sharp focal point",
                        aliases=["body_focus"]
                    ),
                    TagDefinition(
                        name="Background Blur",
                        description="Background intentionally out of focus",
                        aliases=["bokeh", "shallow_dof", "blurred_background"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # LIGHTING - Light conditions and quality
    # =========================================================================
    TagDefinition(
        name="Lighting",
        description="Quality and characteristics of light in the scene",
        children=[
            TagDefinition(
                name="Natural Light",
                description="Illumination from natural sources like sun or sky",
                aliases=["natural_light", "daylight", "sunlight"],
                children=[
                    TagDefinition(
                        name="Golden Hour",
                        description="Warm, soft light shortly after sunrise or before sunset",
                        aliases=["golden_hour", "magic_hour", "warm_light"]
                    ),
                    TagDefinition(
                        name="Overcast",
                        description="Diffused light from cloudy sky",
                        aliases=["overcast", "cloudy", "diffused_light"]
                    ),
                    TagDefinition(
                        name="Harsh Sunlight",
                        description="Direct, strong sunlight creating sharp shadows",
                        aliases=["harsh_light", "direct_sunlight", "high_contrast"]
                    ),
                ],
            ),
            TagDefinition(
                name="Artificial Light",
                description="Illumination from man-made sources",
                aliases=["artificial_light", "studio_lighting"],
                children=[
                    TagDefinition(
                        name="Ring Light",
                        description="Circular light creating even, shadowless illumination",
                        aliases=["ring_light", "catchlight"]
                    ),
                    TagDefinition(
                        name="Neon",
                        description="Colored lighting from neon or LED sources",
                        aliases=["neon", "neon_lights", "colored_lighting"]
                    ),
                ],
            ),
            TagDefinition(
                name="Low Light",
                description="Dim lighting conditions",
                aliases=["low_light", "dim", "dark", "moody_lighting"]
            ),
            TagDefinition(
                name="Backlighting",
                description="Primary light source behind the subject",
                aliases=["backlit", "backlighting", "silhouette", "rim_light"]
            ),
        ],
    ),

    # =========================================================================
    # TIME - Temporal context
    # =========================================================================
    TagDefinition(
        name="Time",
        description="Temporal context of when the scene appears to take place",
        children=[
            TagDefinition(
                name="Time of Day",
                description="Part of the day based on lighting and context",
                children=[
                    TagDefinition(
                        name="Morning",
                        description="Early part of the day",
                        aliases=["morning", "sunrise", "dawn"]
                    ),
                    TagDefinition(
                        name="Daytime",
                        description="Middle part of the day with full daylight",
                        aliases=["day", "daytime", "afternoon", "midday"]
                    ),
                    TagDefinition(
                        name="Evening",
                        description="Late part of the day as sun sets",
                        aliases=["evening", "sunset", "dusk"]
                    ),
                    TagDefinition(
                        name="Night",
                        description="Dark period between sunset and sunrise",
                        aliases=["night", "nighttime", "after_dark"]
                    ),
                ],
            ),
        ],
    ),

    # =========================================================================
    # MOOD - Emotional tone and atmosphere
    # =========================================================================
    TagDefinition(
        name="Mood",
        description="Overall emotional tone or atmosphere of the scene",
        children=[
            TagDefinition(
                name="Playful",
                description="Light-hearted, fun, or teasing atmosphere",
                aliases=["playful", "fun", "teasing", "flirty", "cheeky"]
            ),
            TagDefinition(
                name="Sensual",
                description="Suggesting or expressing physical attraction",
                aliases=["sensual", "seductive", "alluring", "sexy"]
            ),
            TagDefinition(
                name="Romantic",
                description="Expressing love or intimate affection",
                aliases=["romantic", "intimate", "loving"]
            ),
            TagDefinition(
                name="Casual",
                description="Relaxed, everyday, informal atmosphere",
                aliases=["casual", "relaxed", "everyday", "candid"]
            ),
            TagDefinition(
                name="Professional",
                description="Formal, polished, business-like atmosphere",
                aliases=["professional", "formal", "polished"]
            ),
            TagDefinition(
                name="Artistic",
                description="Emphasizing creative or aesthetic qualities",
                aliases=["artistic", "creative", "aesthetic"]
            ),
            TagDefinition(
                name="Energetic",
                description="High energy, dynamic, or exciting atmosphere",
                aliases=["energetic", "dynamic", "exciting", "active"]
            ),
        ],
    ),

    # =========================================================================
    # CONTENT RATING - Content classification
    # =========================================================================
    TagDefinition(
        name="Content Rating",
        description="Classification of content explicitness and appropriateness",
        children=[
            TagDefinition(
                name="SFW",
                description="Safe for work - appropriate for general audiences",
                aliases=["sfw", "safe", "clean"]
            ),
            TagDefinition(
                name="Suggestive",
                description="Mildly provocative but not explicit",
                aliases=["suggestive", "risque", "provocative"]
            ),
            TagDefinition(
                name="NSFW",
                description="Not safe for work - adult content",
                aliases=["nsfw", "adult", "explicit", "mature"]
            ),
        ],
    ),
]


# =============================================================================
# STASH CLIENT
# =============================================================================

class StashTaxonomySeeder:
    """Seeds hierarchical taxonomy into Stash via GraphQL"""

    MUTATION_CREATE_TAG = """
    mutation TagCreate($input: TagCreateInput!) {
        tagCreate(input: $input) {
            id
            name
        }
    }
    """

    QUERY_FIND_TAG = """
    query FindTag($name: String!) {
        findTags(tag_filter: { name: { value: $name, modifier: EQUALS } }) {
            tags {
                id
                name
            }
        }
    }
    """

    def __init__(self, stash_url: str, api_key: Optional[str] = None):
        self.stash_url = stash_url.rstrip("/")
        self.graphql_url = f"{self.stash_url}/graphql"
        self.api_key = api_key
        self.created_tags: Dict[str, str] = {}  # name -> id
        self.stats = {"created": 0, "existing": 0, "failed": 0}

    async def _execute(self, query: str, variables: dict) -> dict:
        """Execute GraphQL query"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["ApiKey"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                headers=headers
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                raise Exception(f"GraphQL error: {result['errors']}")

            return result.get("data", {})

    async def find_tag(self, name: str) -> Optional[str]:
        """Find tag by name, return ID if exists"""
        try:
            data = await self._execute(self.QUERY_FIND_TAG, {"name": name})
            tags = data.get("findTags", {}).get("tags", [])
            if tags:
                return tags[0]["id"]
        except Exception:
            pass
        return None

    async def create_tag(
        self,
        name: str,
        description: str,
        aliases: List[str],
        parent_ids: List[str]
    ) -> Optional[str]:
        """Create a tag, return its ID"""
        input_data = {
            "name": name,
            "description": description,
        }
        if aliases:
            input_data["aliases"] = aliases
        if parent_ids:
            input_data["parent_ids"] = parent_ids

        try:
            data = await self._execute(self.MUTATION_CREATE_TAG, {"input": input_data})
            tag_id = data.get("tagCreate", {}).get("id")
            if tag_id:
                self.stats["created"] += 1
                print(f"  ✓ Created: {name}")
                return tag_id
        except Exception as e:
            if "already exists" in str(e).lower():
                existing_id = await self.find_tag(name)
                if existing_id:
                    self.stats["existing"] += 1
                    print(f"  ○ Exists: {name}")
                    return existing_id
            self.stats["failed"] += 1
            print(f"  ✗ Failed: {name} - {e}")
        return None

    async def seed_tag_recursive(
        self,
        tag_def: TagDefinition,
        parent_ids: List[str],
        depth: int = 0
    ) -> Optional[str]:
        """Recursively create tag and its children"""
        indent = "  " * depth

        # Check if already created in this run
        if tag_def.name in self.created_tags:
            return self.created_tags[tag_def.name]

        # Create this tag
        tag_id = await self.create_tag(
            name=tag_def.name,
            description=tag_def.description,
            aliases=tag_def.aliases,
            parent_ids=parent_ids
        )

        if tag_id:
            self.created_tags[tag_def.name] = tag_id

            # Create children with this tag as parent
            for child in tag_def.children:
                await self.seed_tag_recursive(child, [tag_id], depth + 1)

        return tag_id

    async def seed_all(self, dry_run: bool = False):
        """Seed entire taxonomy"""
        print("\n" + "=" * 60)
        print("STASH TAG TAXONOMY SEEDER")
        print("=" * 60)

        if dry_run:
            print("\n[DRY RUN MODE - No changes will be made]\n")
            self._print_taxonomy_preview(TAXONOMY)
            return

        print(f"\nTarget: {self.stash_url}")
        print(f"Tags to process: {self._count_tags(TAXONOMY)}\n")

        for top_level in TAXONOMY:
            print(f"\n[{top_level.name}]")
            await self.seed_tag_recursive(top_level, [], depth=0)

        print("\n" + "=" * 60)
        print("COMPLETE")
        print("=" * 60)
        print(f"Created: {self.stats['created']}")
        print(f"Existing: {self.stats['existing']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total: {sum(self.stats.values())}")

    def _count_tags(self, tags: List[TagDefinition]) -> int:
        """Count total tags including children"""
        count = 0
        for tag in tags:
            count += 1
            count += self._count_tags(tag.children)
        return count

    def _print_taxonomy_preview(self, tags: List[TagDefinition], depth: int = 0):
        """Print taxonomy structure preview"""
        for tag in tags:
            indent = "  " * depth
            aliases_str = f" [{', '.join(tag.aliases[:3])}{'...' if len(tag.aliases) > 3 else ''}]" if tag.aliases else ""
            print(f"{indent}• {tag.name}{aliases_str}")
            print(f"{indent}  {tag.description[:60]}..." if len(tag.description) > 60 else f"{indent}  {tag.description}")
            self._print_taxonomy_preview(tag.children, depth + 1)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Seed hierarchical tag taxonomy into Stash"
    )
    parser.add_argument(
        "--stash-url",
        default="http://localhost:9999",
        help="Stash instance URL"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Stash API key for authentication"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview taxonomy without creating tags"
    )

    args = parser.parse_args()

    seeder = StashTaxonomySeeder(args.stash_url, args.api_key)
    await seeder.seed_all(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
