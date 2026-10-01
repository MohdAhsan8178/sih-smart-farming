#!/usr/bin/env bash
set -euo pipefail

# scripts/reset_edge_db.sh — Demo Database Reset Script (Task 10.4)
#
# Aborts if any gateway or pipeline processes are running.
# Archives data/edge.db (and -wal/-shm if present) into data/archive/edge_<timestamp>.db*
# rather than deleting, leaving the system ready for a clean fresh database creation
# on the next run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo "SIH Demo Database Reset"
echo "======================================================================"

# 1. Process safety check: abort if gateway or pipeline processes are running
RUNNING_PROCS=0

if pgrep -f "gateway/server.py" > /dev/null 2>&1 || pgrep -f "gateway.server" > /dev/null 2>&1; then
    echo "[ERROR] Gateway process is currently running!" >&2
    RUNNING_PROCS=1
fi

if pgrep -f "edge/pipeline.py" > /dev/null 2>&1 || pgrep -f "edge.pipeline" > /dev/null 2>&1; then
    echo "[ERROR] Pipeline process is currently running!" >&2
    RUNNING_PROCS=1
fi

if [ "$RUNNING_PROCS" -ne 0 ]; then
    echo "[ABORT] Please stop all running gateway and pipeline services before resetting DB." >&2
    exit 1
fi

# 2. Resolve active storage directory via shared Python resolver
RESOLVED_DIR=$(python3 -c "import sys; sys.path.insert(0, '.'); from edge.storage import resolve_data_directory; print(resolve_data_directory()[0])" 2>/dev/null || echo "data")
echo "[INFO] Resolved active data directory: $RESOLVED_DIR"

DB_FILE="$RESOLVED_DIR/edge.db"
WAL_FILE="$RESOLVED_DIR/edge.db-wal"
SHM_FILE="$RESOLVED_DIR/edge.db-shm"

if [ ! -f "$DB_FILE" ] && [ ! -f "$WAL_FILE" ] && [ ! -f "$SHM_FILE" ]; then
    echo "[INFO] No active SQLite database found at $DB_FILE. Nothing to archive."
    echo "[OK] Edge database is ready for fresh initialization on next launch."
    exit 0
fi

# 3. Create archive directory
ARCHIVE_DIR="$RESOLVED_DIR/archive"
mkdir -p "$ARCHIVE_DIR"

TS="$(date -u +%Y%m%d_%H%M%SZ)"
echo "[INFO] Archiving active database with timestamp: $TS"

# 4. Move files to archive
if [ -f "$DB_FILE" ]; then
    ARCHIVE_DB="$ARCHIVE_DIR/edge_${TS}.db"
    mv "$DB_FILE" "$ARCHIVE_DB"
    echo "  [ARCHIVED] $DB_FILE -> $ARCHIVE_DB"
fi

if [ -f "$WAL_FILE" ]; then
    ARCHIVE_WAL="$ARCHIVE_DIR/edge_${TS}.db-wal"
    mv "$WAL_FILE" "$ARCHIVE_WAL"
    echo "  [ARCHIVED] $WAL_FILE -> $ARCHIVE_WAL"
fi

if [ -f "$SHM_FILE" ]; then
    ARCHIVE_SHM="$ARCHIVE_DIR/edge_${TS}.db-shm"
    mv "$SHM_FILE" "$ARCHIVE_SHM"
    echo "  [ARCHIVED] $SHM_FILE -> $ARCHIVE_SHM"
fi

echo "======================================================================"
echo "[SUCCESS] Active database successfully archived."
echo "Fresh database schema will be automatically created on next pipeline / gateway run."
echo "======================================================================"
