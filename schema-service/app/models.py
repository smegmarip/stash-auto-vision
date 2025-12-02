"""
Schema Service - Data Models
Pydantic models for request/response validation
"""

import io

from pydantic import BaseModel, Field
from typing import Optional
from fastapi.responses import Response
from ruamel.yaml import YAML


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    service: str
    version: str
    message: Optional[str] = None


class YAMLResponse(Response):
    """Custom YAML response class for FastAPI"""

    media_type = "text/yaml"

    def render(self, content: any) -> bytes:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.width = 120
        stream = io.BytesIO()
        yaml.dump(content, stream)
        return stream.getvalue()


class SchemaResponse(BaseModel):
    """OpenAPI schema response"""

    openapi: str
    info: dict
    paths: dict
    components: Optional[dict] = Field(default_factory=lambda: {"schemas": {}, "securitySchemes": {}})
    security: Optional[list] = Field(default_factory=list)
    tags: Optional[list] = Field(default_factory=list)
    servers: Optional[list] = Field(default_factory=list)
