#!/usr/bin/env python3
"""Fix the 6 tags that failed due to alias conflicts"""

import asyncio
import httpx

STASH_URL = "http://localhost:9999/graphql"

FIXES = [
    {
        "name": "Panties",
        "description": "Underwear covering the pelvic region",
        "aliases": ["panties", "briefs", "thong"],
        "parent_name": "Underwear"
    },
    {
        "name": "Car",
        "description": "Inside or around a personal automobile",
        "aliases": ["car", "automobile", "car_interior"],
        "parent_name": "Vehicle"
    },
    {
        "name": "First Person POV",
        "description": "Shot from the viewpoint of a participant",
        "aliases": ["first_person", "subjective_shot"],
        "parent_name": "POV"
    },
    {
        "name": "Selfie",
        "description": "Self-portrait typically with phone at arm's length",
        "aliases": ["self_shot", "mirror_selfie"],
        "parent_name": "POV"
    },
    {
        "name": "Face Focus",
        "description": "Face is the sharp focal point",
        "aliases": ["portrait", "face_shot"],
        "parent_name": "Focus"
    },
    {
        "name": "NSFW",
        "description": "Not safe for work - adult content",
        "aliases": ["nsfw", "adult", "explicit", "x_rated"],
        "parent_name": "Content Rating"
    },
]

FIND_TAG = """
query FindTag($name: String!) {
    findTags(tag_filter: { name: { value: $name, modifier: EQUALS } }) {
        tags { id name }
    }
}
"""

CREATE_TAG = """
mutation TagCreate($input: TagCreateInput!) {
    tagCreate(input: $input) { id name }
}
"""


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        for fix in FIXES:
            # Find parent tag
            resp = await client.post(STASH_URL, json={
                "query": FIND_TAG,
                "variables": {"name": fix["parent_name"]}
            })
            data = resp.json()
            tags = data.get("data", {}).get("findTags", {}).get("tags", [])

            if not tags:
                print(f"✗ {fix['name']}: Parent '{fix['parent_name']}' not found")
                continue

            parent_id = tags[0]["id"]

            # Create the tag
            resp = await client.post(STASH_URL, json={
                "query": CREATE_TAG,
                "variables": {
                    "input": {
                        "name": fix["name"],
                        "description": fix["description"],
                        "aliases": fix["aliases"],
                        "parent_ids": [parent_id]
                    }
                }
            })
            result = resp.json()

            if "errors" in result:
                print(f"✗ {fix['name']}: {result['errors'][0]['message']}")
            else:
                print(f"✓ Created: {fix['name']} (parent: {fix['parent_name']})")


if __name__ == "__main__":
    asyncio.run(main())
