"""
Schema Service combining multiple OpenAPI schemas into one.
FastAPI server for serving combined OpenAPI schema in JSON and YAML formats.
"""

import os
import yaml
import json
import httpx
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pathlib import Path

from .models import HealthResponse, SchemaResponse, YAMLResponse
from .cache_manager import CacheManager

# Environment configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


CONFIG_PATH = "/config/config.yaml"
TEMPLATE_PATH = "/config/template.yaml"

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global instances
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Initialize services on startup, cleanup on shutdown
    """
    global cache_manager

    logger.info("Starting Schema Service...")

    # Initialize cache manager
    cache_manager = CacheManager(providerFn=generate_openapi_schema, refresh_interval=300)
    cache_manager.start()

    # override FastAPI generator
    app.openapi = get_cached_schema
    yield

    # Cleanup
    logger.info("Shutting down Schema Service...")

    if cache_manager:
        await cache_manager.stop()

    logger.info("Schema Service stopped")


app = FastAPI(
    title="Vision API Schema",
    description="Combined OpenAPI schema for Vision API services",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/vision_openapi.json",
)


async def generate_openapi_schema():
    """
    Generate combined OpenAPI schema from configured services
    Returns:
        dict: Combined OpenAPI schema
    """
    try:
        # Validate config exists
        if not os.path.exists(CONFIG_PATH) or not os.path.exists(TEMPLATE_PATH):
            raise RuntimeError(f"Schema config or template not found: {CONFIG_PATH} / {TEMPLATE_PATH}")

        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        with open(TEMPLATE_PATH) as f:
            combined = yaml.safe_load(f)

        combined["paths"] = {}
        combined["servers"] = []
        combined["tags"] = []
        combined["components"] = {"schemas": {}, "securitySchemes": {}}

        for svr in config["servers"]:
            tags = None
            try:
                if "x-container-url" in svr:
                    # merge remote schemas
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"{svr['x-container-url'].rstrip('/')}/openapi.json", timeout=10)
                        data = resp.json()

                    namespace = svr.get("x-namespace", "unknown")
                    port = svr["x-container-url"].split(":")[-1]
                    tags = list(filter(lambda tag: tag.get("x-port") == int(port), config["tags"]))
                    # filter tag properties to name/description only
                    health_tag = (list(filter(lambda tag: tag["name"] == "health", config["tags"])) or [None])[0]
                    svc_tags = list([dict({k: t[k] for k in t if not k.startswith("x-")}) for t in tags])

                    if data:
                        paths = data.get("paths", {})
                        if len(paths) > 0:
                            if tags and len(tags) > 0:
                                for route in paths:
                                    for method in paths[route]:
                                        # add tags to each path method
                                        paths[route][method]["tags"] = [tag.get("name") for tag in svc_tags]
                                if health_tag and f"/{namespace}/health" in paths:
                                    svc_tags.append(health_tag)  # will deduplicate later
                                    for method in paths[f"/{namespace}/health"]:
                                        if "tags" in paths[f"/{namespace}/health"][method]:
                                            paths[f"/{namespace}/health"][method]["tags"].append("health")
                                        else:
                                            paths[f"/{namespace}/health"][method]["tags"] = ["health"]

                        combined["paths"].update(paths)
                        combined["servers"].append(dict({k: svr[k] for k in svr if not k.startswith("x-")}))
                        combined["tags"].extend(svc_tags)
                        combined["components"]["schemas"].update(data.get("components", {}).get("schemas", {}))
                        combined["components"]["securitySchemes"].update(
                            data.get("components", {}).get("securitySchemes", {})
                        )
            except Exception as e:
                logger.warning(f"Unable to fetch schema for service {svr.get('x-container-url', 'unknown')}: {e}")
                continue

        # Deduplicate servers and tags
        combined["servers"] = dedupe_list(combined["servers"])
        combined["tags"] = dedupe_list(combined["tags"])
        return combined
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def dedupe_list(data):
    """Deduplicate list of dicts while preserving order."""
    seen = set()
    out = []
    for item in data:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def get_cached_schema() -> SchemaResponse:
    """Get cached OpenAPI schema synchronously if available."""
    global cache_manager
    if cache_manager:
        return cache_manager.get()
    return SchemaResponse(
        openapi="3.1.0",
        info={
            "title": "Vision API (initializing…)",
            "description": "Vision Services API.",
            "version": "1.0.0",
        },
        paths={},
    )


@app.get("/openapi.json", response_model=SchemaResponse, response_class=JSONResponse)
async def combined_json():
    """Get combined OpenAPI schema in JSON format."""
    json_schema = get_cached_schema()
    if json_schema is None:
        json_schema = await generate_openapi_schema()
    return json_schema


@app.get("/openapi.yaml", response_model=SchemaResponse, response_class=YAMLResponse)
@app.get("/openapi.yml", response_model=SchemaResponse, response_class=YAMLResponse)
async def combined_yaml():
    """Get combined OpenAPI schema in YAML format."""
    json_schema = await combined_json()
    return json_schema


@app.get("/schema/health", response_model=HealthResponse, response_class=JSONResponse)
async def health_check():
    """Health check endpoint"""

    return HealthResponse(
        status="healthy",
        service="schema-service",
        version="1.0.0",
        message="Schema service active",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5009)
