"""
Captioning Service - Stash GraphQL Client
Client for interacting with Stash's GraphQL API for tag management
"""

import httpx
from typing import List, Optional, Dict, Any
import logging

from .models import TagTaxonomyNode

logger = logging.getLogger(__name__)


# GraphQL queries and mutations
QUERY_ALL_TAGS = """
query AllTags {
    allTags {
        id
        name
        aliases
        parent_count
        child_count
        parents {
            id
            name
        }
        children {
            id
            name
        }
    }
}
"""

QUERY_TAG_BY_NAME = """
query FindTag($name: String!) {
    findTags(tag_filter: { name: { value: $name, modifier: EQUALS } }) {
        count
        tags {
            id
            name
            aliases
            parents { id name }
            children { id name }
        }
    }
}
"""

MUTATION_CREATE_TAG = """
mutation TagCreate($input: TagCreateInput!) {
    tagCreate(input: $input) {
        id
        name
        aliases
    }
}
"""

MUTATION_UPDATE_TAG = """
mutation TagUpdate($input: TagUpdateInput!) {
    tagUpdate(input: $input) {
        id
        name
        aliases
        parents { id name }
        children { id name }
    }
}
"""


class StashClient:
    """Client for Stash GraphQL API"""

    def __init__(
        self,
        stash_url: str = "http://localhost:9999",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Stash client

        Args:
            stash_url: Base URL of Stash instance
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.stash_url = stash_url.rstrip("/")
        self.graphql_url = f"{self.stash_url}/graphql"
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["ApiKey"] = self.api_key

            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout
            )
        return self._client

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute GraphQL query/mutation

        Args:
            query: GraphQL query string
            variables: Optional variables dict

        Returns:
            Response data dict

        Raises:
            Exception on GraphQL errors
        """
        client = await self._get_client()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = await client.post(self.graphql_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                errors = result["errors"]
                error_msg = "; ".join(e.get("message", str(e)) for e in errors)
                raise Exception(f"GraphQL errors: {error_msg}")

            return result.get("data", {})

        except httpx.HTTPStatusError as e:
            logger.error(f"Stash API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing GraphQL: {e}")
            raise

    async def get_all_tags(self) -> List[TagTaxonomyNode]:
        """
        Fetch all tags from Stash

        Returns:
            List of TagTaxonomyNode objects
        """
        logger.info("Fetching all tags from Stash...")
        data = await self._execute(QUERY_ALL_TAGS)

        tags = []
        all_tags = data.get("allTags", [])

        for tag_data in all_tags:
            node = TagTaxonomyNode(
                id=tag_data["id"],
                name=tag_data["name"],
                aliases=tag_data.get("aliases", []) or [],
                parent_id=tag_data["parents"][0]["id"] if tag_data.get("parents") else None,
                children=[c["id"] for c in tag_data.get("children", [])],
                category=self._infer_category(tag_data)
            )
            tags.append(node)

        logger.info(f"Fetched {len(tags)} tags from Stash")
        return tags

    def _infer_category(self, tag_data: Dict[str, Any]) -> Optional[str]:
        """
        Infer category from tag hierarchy

        If the tag has a parent, use the parent's name as category
        """
        parents = tag_data.get("parents", [])
        if parents:
            return parents[0]["name"].lower()
        return None

    async def find_tag(self, name: str) -> Optional[TagTaxonomyNode]:
        """
        Find a tag by exact name

        Args:
            name: Tag name to search for

        Returns:
            TagTaxonomyNode if found, None otherwise
        """
        data = await self._execute(QUERY_TAG_BY_NAME, {"name": name})

        result = data.get("findTags", {})
        tags = result.get("tags", [])

        if not tags:
            return None

        tag_data = tags[0]
        return TagTaxonomyNode(
            id=tag_data["id"],
            name=tag_data["name"],
            aliases=tag_data.get("aliases", []) or [],
            parent_id=tag_data["parents"][0]["id"] if tag_data.get("parents") else None,
            children=[c["id"] for c in tag_data.get("children", [])],
        )

    async def create_tag(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        child_ids: Optional[List[str]] = None
    ) -> TagTaxonomyNode:
        """
        Create a new tag in Stash

        Args:
            name: Tag name
            aliases: Optional list of aliases
            parent_ids: Optional parent tag IDs
            child_ids: Optional child tag IDs

        Returns:
            Created TagTaxonomyNode
        """
        input_data: Dict[str, Any] = {"name": name}

        if aliases:
            input_data["aliases"] = aliases
        if parent_ids:
            input_data["parent_ids"] = parent_ids
        if child_ids:
            input_data["child_ids"] = child_ids

        data = await self._execute(MUTATION_CREATE_TAG, {"input": input_data})
        tag_data = data["tagCreate"]

        logger.info(f"Created tag: {name} (id: {tag_data['id']})")

        return TagTaxonomyNode(
            id=tag_data["id"],
            name=tag_data["name"],
            aliases=tag_data.get("aliases", []) or [],
            parent_id=parent_ids[0] if parent_ids else None,
            children=child_ids or []
        )

    async def update_tag(
        self,
        tag_id: str,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        child_ids: Optional[List[str]] = None
    ) -> TagTaxonomyNode:
        """
        Update an existing tag in Stash

        Args:
            tag_id: Tag ID to update
            name: New name (optional)
            aliases: New aliases (optional)
            parent_ids: New parent IDs (optional)
            child_ids: New child IDs (optional)

        Returns:
            Updated TagTaxonomyNode
        """
        input_data: Dict[str, Any] = {"id": tag_id}

        if name is not None:
            input_data["name"] = name
        if aliases is not None:
            input_data["aliases"] = aliases
        if parent_ids is not None:
            input_data["parent_ids"] = parent_ids
        if child_ids is not None:
            input_data["child_ids"] = child_ids

        data = await self._execute(MUTATION_UPDATE_TAG, {"input": input_data})
        tag_data = data["tagUpdate"]

        return TagTaxonomyNode(
            id=tag_data["id"],
            name=tag_data["name"],
            aliases=tag_data.get("aliases", []) or [],
            parent_id=tag_data["parents"][0]["id"] if tag_data.get("parents") else None,
            children=[c["id"] for c in tag_data.get("children", [])]
        )

    async def ensure_tag_exists(
        self,
        name: str,
        aliases: Optional[List[str]] = None,
        parent_name: Optional[str] = None
    ) -> TagTaxonomyNode:
        """
        Ensure a tag exists, creating it if necessary

        Args:
            name: Tag name
            aliases: Optional aliases
            parent_name: Optional parent tag name (must already exist)

        Returns:
            Existing or newly created TagTaxonomyNode
        """
        existing = await self.find_tag(name)
        if existing:
            return existing

        parent_ids = None
        if parent_name:
            parent = await self.find_tag(parent_name)
            if parent:
                parent_ids = [parent.id]

        return await self.create_tag(name, aliases, parent_ids)

    async def health_check(self) -> bool:
        """
        Check if Stash API is reachable

        Returns:
            True if healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.stash_url}/")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Stash health check failed: {e}")
            return False
