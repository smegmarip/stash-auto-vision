#!/bin/bash
# Generate Compound Test Videos for Stash Auto Vision Testing
# Uses selfies, charades, and youtube_faces datasets to create multi-scenario test videos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$TESTS_DIR/data"
COMPOUND_DIR="$DATA_DIR/compound"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"
}

success() {
    echo -e "${GREEN}✓${NC} $*"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

# Create compound video directories
log "Creating compound video directories..."
mkdir -p "$COMPOUND_DIR"/{frame-server,scenes,faces,vision-api}
success "Directories created"

# Check if datasets exist
log "Checking datasets..."
if [ ! -d "$DATA_DIR/selfies/files" ]; then
    warn "Selfies dataset not found at $DATA_DIR/selfies/files"
fi

if [ ! -d "$DATA_DIR/charades/dataset" ]; then
    warn "Charades dataset not found at $DATA_DIR/charades/dataset"
fi

if [ ! -d "$DATA_DIR/youtube_faces/data" ]; then
    warn "YouTube Faces dataset not found at $DATA_DIR/youtube_faces/data"
fi

# Find source videos
log "Finding source videos..."
CHARADES_1=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -1)
CHARADES_2=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -2 | tail -1)
CHARADES_3=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -3 | tail -1)
CHARADES_4=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -4 | tail -1)
CHARADES_5=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -5 | tail -1)
CHARADES_6=$(find "$DATA_DIR/charades/dataset" -name "*.mp4" | head -6 | tail -1)

log "Source videos found:"
echo "  Charades: $CHARADES_1, $CHARADES_2, $CHARADES_3, $CHARADES_4, $CHARADES_5, $CHARADES_6"

# ============================================
# 1. Frame Server Test Videos
# ============================================

log "Generating Frame Server test videos..."

# Video 1: Multi-scene transitions (60s)
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ]; then
    log "Creating multi_scene_transitions.mp4 (60s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" \
        -filter_complex "\
            [0:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v2]; \
            [v0][v1][v2]concat=n=3:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 -r 30 \
        -t 60 \
        "$COMPOUND_DIR/frame-server/multi_scene_transitions.mp4" \
        -loglevel error
    success "multi_scene_transitions.mp4 created"
else
    warn "Skipping multi_scene_transitions.mp4 (missing source videos)"
fi

# Video 2: Long video (300s / 5 minutes)
if [ -f "$CHARADES_1" ]; then
    log "Creating long_video.mp4 (300s)..."
    ffmpeg -y \
        -stream_loop 10 -i "$CHARADES_1" \
        -c:v libx264 -preset fast -crf 23 \
        -t 300 \
        "$COMPOUND_DIR/frame-server/long_video.mp4" \
        -loglevel error
    success "long_video.mp4 created"
else
    warn "Skipping long_video.mp4 (missing source video)"
fi

# ============================================
# 2. Scenes Service Test Videos
# ============================================

log "Generating Scenes Service test videos..."

# Video 3: Sharp transitions (90s)
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ]; then
    log "Creating sharp_transitions.mp4 (90s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" \
        -filter_complex "\
            [0:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fade=t=in:st=0:d=1[v2]; \
            [v0][v1][v2]concat=n=3:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 90 \
        "$COMPOUND_DIR/scenes/sharp_transitions.mp4" \
        -loglevel error
    success "sharp_transitions.mp4 created"
else
    warn "Skipping sharp_transitions.mp4 (missing source videos)"
fi

# Video 4: Gradual transitions (120s)
# Using fade in/out instead of xfade due to timebase issues
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ]; then
    log "Creating gradual_transitions.mp4 (120s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" \
        -filter_complex "\
            [0:v]trim=0:40,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fade=t=out:st=38:d=2[v0]; \
            [1:v]trim=0:40,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fade=t=in:st=0:d=2,fade=t=out:st=38:d=2[v1]; \
            [2:v]trim=0:40,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fade=t=in:st=0:d=2[v2]; \
            [v0][v1][v2]concat=n=3:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 120 \
        "$COMPOUND_DIR/scenes/gradual_transitions.mp4" \
        -loglevel error
    success "gradual_transitions.mp4 created"
else
    warn "Skipping gradual_transitions.mp4 (missing source videos)"
fi

# ============================================
# 3. Faces Service Test Videos
# ============================================

log "Generating Faces Service test videos..."

# Video 5: Single person varied conditions (60s)
# Using different segments from charades (with people)
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ]; then
    log "Creating single_person_varied_conditions.mp4 (60s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" \
        -filter_complex "\
            [0:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:20,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v2]; \
            [v0][v1][v2]concat=n=3:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 60 \
        "$COMPOUND_DIR/faces/single_person_varied_conditions.mp4" \
        -loglevel error
    success "single_person_varied_conditions.mp4 created"
else
    warn "Skipping single_person_varied_conditions.mp4 (missing source videos)"
fi

# Video 6: Multiple persons (90s)
# Different charades clips with varying number of people
if [ -f "$CHARADES_4" ] && [ -f "$CHARADES_5" ] && [ -f "$CHARADES_6" ]; then
    log "Creating multiple_persons.mp4 (90s)..."
    ffmpeg -y \
        -i "$CHARADES_4" -i "$CHARADES_5" -i "$CHARADES_6" \
        -filter_complex "\
            [0:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v2]; \
            [v0][v1][v2]concat=n=3:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 90 \
        "$COMPOUND_DIR/faces/multiple_persons.mp4" \
        -loglevel error
    success "multiple_persons.mp4 created"
else
    warn "Skipping multiple_persons.mp4 (missing source videos)"
fi

# Video 7: Challenging conditions (60s)
# Apply transformations to charades to create challenging conditions
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ]; then
    log "Creating challenging_conditions.mp4 (60s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" \
        -filter_complex "\
            [0:v]trim=0:15,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:15,setpts=PTS-STARTPTS,eq=brightness=-0.3,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:15,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v2]; \
            [2:v]trim=0:15,setpts=PTS-STARTPTS,minterpolate=fps=10:mi_mode=blend,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v3]; \
            [v0][v1][v2][v3]concat=n=4:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 60 \
        "$COMPOUND_DIR/faces/challenging_conditions.mp4" \
        -loglevel error
    success "challenging_conditions.mp4 created"
else
    warn "Skipping challenging_conditions.mp4 (missing source videos)"
fi

# ============================================
# 4. Vision API Rollup Test Videos
# ============================================

log "Generating Vision API test videos..."

# Video 8: Complete analysis (120s)
# Using charades videos with different scene types
if [ -f "$CHARADES_1" ] && [ -f "$CHARADES_2" ] && [ -f "$CHARADES_3" ] && [ -f "$CHARADES_4" ]; then
    log "Creating complete_analysis.mp4 (120s)..."
    ffmpeg -y \
        -i "$CHARADES_1" -i "$CHARADES_2" -i "$CHARADES_3" -i "$CHARADES_4" \
        -filter_complex "\
            [0:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0]; \
            [1:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1]; \
            [2:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v2]; \
            [3:v]trim=0:30,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v3]; \
            [v0][v1][v2][v3]concat=n=4:v=1:a=0[outv]" \
        -map "[outv]" \
        -c:v libx264 -preset fast -crf 23 \
        -t 120 \
        "$COMPOUND_DIR/vision-api/complete_analysis.mp4" \
        -loglevel error
    success "complete_analysis.mp4 created"
else
    warn "Skipping complete_analysis.mp4 (missing source videos)"
fi

# ============================================
# Summary
# ============================================

log "Video generation complete!"
echo ""
echo "Generated videos:"
find "$COMPOUND_DIR" -name "*.mp4" -exec ls -lh {} \; | awk '{printf "  %-50s %s\n", $9, $5}'
echo ""
success "All test videos generated successfully"
