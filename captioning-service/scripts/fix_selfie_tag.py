#!/usr/bin/env python3
"""Fix the Selfie tag by first removing it from Using Phone aliases"""

import asyncio
import httpx

STASH_URL = "http://localhost:9999/graphql"

FIND_TAG = """
query FindTag($name: String!) {
    findTags(tag_filter: { name: { value: $name, modifier: EQUALS } }) {
        tags { id name aliases }
    }
}
"""

UPDATE_TAG = """
mutation TagUpdate($input: TagUpdateInput!) {
    tagUpdate(input: $input) { id name aliases }
}
"""

CREATE_TAG = """
mutation TagCreate($input: TagCreateInput!) {
    tagCreate(input: $input) { id name }
}
"""


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Find "Using Phone" and get current aliases
        resp = await client.post(STASH_URL, json={
            "query": FIND_TAG,
            "variables": {"name": "Using Phone"}
        })
        data = resp.json()
        tag = data["data"]["findTags"]["tags"][0]
        tag_id = tag["id"]
        old_aliases = tag["aliases"]
        new_aliases = [a for a in old_aliases if a.lower() != "selfie"]

        print(f"Using Phone current aliases: {old_aliases}")
        print(f"Removing 'selfie', new aliases: {new_aliases}")

        # Update to remove selfie alias
        resp = await client.post(STASH_URL, json={
            "query": UPDATE_TAG,
            "variables": {"input": {"id": tag_id, "aliases": new_aliases}}
        })
        result = resp.json()
        if "errors" in result:
            print(f"✗ Failed to update Using Phone: {result['errors']}")
            return
        print(f"✓ Updated 'Using Phone' aliases")

        # Find POV parent
        resp = await client.post(STASH_URL, json={
            "query": FIND_TAG,
            "variables": {"name": "POV"}
        })
        pov_id = resp.json()["data"]["findTags"]["tags"][0]["id"]

        # Now create Selfie with the selfie alias
        resp = await client.post(STASH_URL, json={
            "query": CREATE_TAG,
            "variables": {"input": {
                "name": "Selfie",
                "description": "Self-portrait typically with phone at arm's length",
                "aliases": ["selfie", "self_shot", "mirror_selfie"],
                "parent_ids": [pov_id]
            }}
        })
        result = resp.json()
        if "errors" in result:
            print(f"✗ Selfie: {result['errors'][0]['message']}")
        else:
            print(f"✓ Created: Selfie (parent: POV)")


if __name__ == "__main__":
    asyncio.run(main())
