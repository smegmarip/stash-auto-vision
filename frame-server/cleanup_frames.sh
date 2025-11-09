#!/bin/bash
#
# Frame Cleanup Script
# Removes extracted frames older than configured TTL
#
# Run via cron: 0 * * * * /app/cleanup_frames.sh
#

set -e

# Configuration
FRAME_DIR="${FRAME_DIR:-/tmp/frames}"
TTL_HOURS="${FRAME_TTL_HOURS:-2}"
LOG_FILE="${LOG_FILE:-/var/log/frame-cleanup.log}"
DRY_RUN="${DRY_RUN:-false}"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting frame cleanup (TTL: ${TTL_HOURS}h, Dir: ${FRAME_DIR})"

# Check if frame directory exists
if [ ! -d "$FRAME_DIR" ]; then
    log "Frame directory does not exist: $FRAME_DIR"
    exit 0
fi

# Count frames before cleanup
TOTAL_DIRS_BEFORE=$(find "$FRAME_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
TOTAL_FILES_BEFORE=$(find "$FRAME_DIR" -type f -name "*.jpg" -o -name "*.png" | wc -l)
DISK_BEFORE=$(du -sh "$FRAME_DIR" 2>/dev/null | cut -f1)

log "Before cleanup: $TOTAL_DIRS_BEFORE directories, $TOTAL_FILES_BEFORE files, $DISK_BEFORE total"

# Find and remove directories older than TTL
# Using -mmin for better precision (TTL_HOURS * 60 minutes)
TTL_MINUTES=$((TTL_HOURS * 60))

if [ "$DRY_RUN" = "true" ]; then
    log "DRY RUN: Would delete the following directories:"
    find "$FRAME_DIR" -mindepth 1 -maxdepth 1 -type d -mmin +${TTL_MINUTES} -print
else
    DELETED_COUNT=0

    while IFS= read -r dir; do
        if [ -d "$dir" ]; then
            DIR_SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            log "Deleting: $dir (size: $DIR_SIZE, age: >$TTL_HOURS hours)"
            rm -rf "$dir"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        fi
    done < <(find "$FRAME_DIR" -mindepth 1 -maxdepth 1 -type d -mmin +${TTL_MINUTES})

    log "Deleted $DELETED_COUNT directories"
fi

# Remove empty parent directories
find "$FRAME_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

# Count frames after cleanup
TOTAL_DIRS_AFTER=$(find "$FRAME_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
TOTAL_FILES_AFTER=$(find "$FRAME_DIR" -type f -name "*.jpg" -o -name "*.png" | wc -l)
DISK_AFTER=$(du -sh "$FRAME_DIR" 2>/dev/null | cut -f1)

log "After cleanup: $TOTAL_DIRS_AFTER directories, $TOTAL_FILES_AFTER files, $DISK_AFTER total"

# Calculate freed space (approximate)
DIRS_FREED=$((TOTAL_DIRS_BEFORE - TOTAL_DIRS_AFTER))
FILES_FREED=$((TOTAL_FILES_BEFORE - TOTAL_FILES_AFTER))

if [ $DIRS_FREED -gt 0 ] || [ $FILES_FREED -gt 0 ]; then
    log "Cleanup complete: Freed $DIRS_FREED directories, $FILES_FREED files"
else
    log "No frames to clean up"
fi

exit 0
