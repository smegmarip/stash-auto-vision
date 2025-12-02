#!/bin/bash
# Fetch OpenAPI schemas from all FastAPI services
# Usage: ./scripts/fetch-openapi-schemas.sh

SERVICES=(
  "vision-api:5010"
  "frame-server:5001"
  "scenes-service:5002"
  "faces-service:5003"
  "semantics-service:5004"
)

OUTPUT_DIR="schemas"
mkdir -p "$OUTPUT_DIR"

echo "Fetching OpenAPI schemas from FastAPI services..."
echo ""

for service in "${SERVICES[@]}"; do
  name="${service%%:*}"
  port="${service##*:}"
  echo "Fetching $name schema from port $port..."
  curl -s "http://localhost:$port/openapi.json" > "$OUTPUT_DIR/$name.json" 2>/dev/null
  if [ $? -eq 0 ] && [ -s "$OUTPUT_DIR/$name.json" ]; then
    # Validate it's valid JSON
    if python3 -c "import json; json.load(open('$OUTPUT_DIR/$name.json'))" 2>/dev/null; then
      echo "  ✓ Saved to $OUTPUT_DIR/$name.json"
    else
      echo "  ✗ Invalid JSON response from $name"
      rm -f "$OUTPUT_DIR/$name.json"
    fi
  else
    echo "  ✗ Failed to fetch $name schema (service may not be running)"
    rm -f "$OUTPUT_DIR/$name.json"
  fi
done

echo ""
echo "Schemas saved to $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No schemas fetched."
