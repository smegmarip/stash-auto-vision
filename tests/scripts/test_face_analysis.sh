#!/bin/bash

# Test face analysis on a sample video
VIDEO_PATH="/media/videos/charades/dataset/001YG.mp4"
SCENE_ID="test_001"

echo "=== Testing Face Analysis ==="
echo "Video: $VIDEO_PATH"
echo ""

# Submit job
echo "1. Submitting face analysis job..."
RESPONSE=$(curl -s -X POST http://localhost:5010/vision/analyze/faces \
  -H "Content-Type: application/json" \
  -d "{
    \"video_path\": \"$VIDEO_PATH\",
    \"scene_id\": \"$SCENE_ID\",
    \"parameters\": {
      \"min_confidence\": 0.8,
      \"max_faces\": 10,
      \"sampling_interval\": 1.0,
      \"enable_deduplication\": true
    }
  }")

echo "$RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo ""
echo "Job ID: $JOB_ID"
echo ""

# Poll status
echo "2. Polling job status..."
while true; do
  STATUS_RESPONSE=$(curl -s http://localhost:5010/vision/analyze/faces/jobs/$JOB_ID/status)
  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  PROGRESS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('progress', 0))")
  
  echo "Status: $STATUS | Progress: $(python3 -c "print(f'{float('$PROGRESS') * 100:.1f}%')")"
  
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    echo ""
    echo "$STATUS_RESPONSE" | python3 -m json.tool
    break
  fi
  
  sleep 2
done

# Get results if completed
if [ "$STATUS" = "completed" ]; then
  echo ""
  echo "3. Fetching results..."
  curl -s http://localhost:5010/vision/analyze/faces/jobs/$JOB_ID/results | python3 -m json.tool
fi
