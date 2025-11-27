#!/bin/bash
#
# Download Cleanup Script
# Removes downloaded files older than CACHE_TTL
#
# Run via cron: 0 * * * * /app/cleanup_downloads.sh
#

set -e

# Configuration
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/downloads}"
CACHE_TTL="${CACHE_TTL:-31536000}"  # Default: 1 year in seconds
TTL_HOURS=$((CACHE_TTL / 3600))     # Convert to hours
LOG_FILE="${LOG_FILE:-/var/log/download-cleanup.log}"
DRY_RUN="${DRY_RUN:-false}"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting download cleanup (TTL: ${TTL_HOURS}h from CACHE_TTL=${CACHE_TTL}s, Dir: ${DOWNLOAD_DIR})"

# Check if download directory exists
if [ ! -d "$DOWNLOAD_DIR" ]; then
    log "Download directory does not exist: $DOWNLOAD_DIR"
    exit 0
fi

# Count files before cleanup
TOTAL_FILES_BEFORE=$(find "$DOWNLOAD_DIR" -type f | wc -l)
DISK_BEFORE=$(du -sh "$DOWNLOAD_DIR" 2>/dev/null | cut -f1)

log "Before cleanup: $TOTAL_FILES_BEFORE files, $DISK_BEFORE total"

# Find and remove files older than TTL
# Using -mmin for better precision (TTL_HOURS * 60 minutes)
TTL_MINUTES=$((TTL_HOURS * 60))

if [ "$DRY_RUN" = "true" ]; then
    log "DRY RUN: Would delete the following files:"
    find "$DOWNLOAD_DIR" -type f -mmin +${TTL_MINUTES} -print
else
    DELETED_COUNT=0

    while IFS= read -r file; do
        if [ -f "$file" ]; then
            FILE_SIZE=$(du -h "$file" 2>/dev/null | cut -f1)
            log "Deleting: $file (size: $FILE_SIZE, age: >$TTL_HOURS hours)"
            rm -f "$file"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        fi
    done < <(find "$DOWNLOAD_DIR" -type f -mmin +${TTL_MINUTES})

    log "Deleted $DELETED_COUNT files"
fi

# Remove empty directories
find "$DOWNLOAD_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

# Count files after cleanup
TOTAL_FILES_AFTER=$(find "$DOWNLOAD_DIR" -type f | wc -l)
DISK_AFTER=$(du -sh "$DOWNLOAD_DIR" 2>/dev/null | cut -f1)

log "After cleanup: $TOTAL_FILES_AFTER files, $DISK_AFTER total"

# Calculate freed space (approximate)
FILES_FREED=$((TOTAL_FILES_BEFORE - TOTAL_FILES_AFTER))

if [ $FILES_FREED -gt 0 ]; then
    log "Cleanup complete: Freed $FILES_FREED files"
else
    log "No downloads to clean up"
fi

exit 0
