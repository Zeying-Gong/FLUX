#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# kill_all.sh
#
# One-stop kill for all flux distributed data-generation runs.
# - Stops all flux_formal_* docker containers
# - Kills all run_distributed.sh and their child processes
# - Kills any orphaned tracking_episode_collection_camera_only.py
# ──────────────────────────────────────────────────────────────────────
set -e

echo "[kill_all] Stopping flux_formal_* docker containers..."
CONTAINERS=$(docker ps -q --filter name=flux_formal 2>/dev/null || true)
if [ -n "$CONTAINERS" ]; then
  docker stop $CONTAINERS
  echo "[kill_all] Stopped containers."
else
  echo "[kill_all] No flux_formal containers found."
fi

echo "[kill_all] Killing run_distributed.sh processes (and children)..."
RDS_PIDS=$(pgrep -f run_distributed.sh 2>/dev/null || true)
if [ -n "$RDS_PIDS" ]; then
  # Kill children first, then parents
  for pid in $RDS_PIDS; do
    CHILD_PIDS=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$CHILD_PIDS" ]; then
      kill -TERM $CHILD_PIDS 2>/dev/null || true
    fi
    kill -TERM "$pid" 2>/dev/null || true
  done
  echo "[kill_all] Killed run_distributed.sh PIDs: $RDS_PIDS"
else
  echo "[kill_all] No run_distributed.sh processes found."
fi

echo "[kill_all] Cleaning up any orphaned python tracking processes outside docker..."
ORPHAN_PIDS=$(pgrep -f tracking_episode_collection_camera_only 2>/dev/null || true)
if [ -n "$ORPHAN_PIDS" ]; then
  kill -TERM $ORPHAN_PIDS 2>/dev/null || true
  echo "[kill_all] Killed orphan PIDs: $ORPHAN_PIDS"
else
  echo "[kill_all] No orphan tracking processes found."
fi

sleep 2

echo "[kill_all] Verifying no flux processes remain..."
REMAINING_CONTAINERS=$(docker ps -q --filter name=flux_formal 2>/dev/null || true)
if [ -n "$REMAINING_CONTAINERS" ]; then
  echo "[kill_all] WARNING: Some containers still running, force killing..."
  docker kill $REMAINING_CONTAINERS 2>/dev/null || true
fi

if pgrep -f run_distributed.sh >/dev/null 2>&1; then
  echo "[kill_all] WARNING: run_distributed.sh still alive, sending SIGKILL..."
  pkill -9 -f run_distributed.sh 2>/dev/null || true
fi

if pgrep -f tracking_episode_collection_camera_only >/dev/null 2>&1; then
  echo "[kill_all] WARNING: orphan python still alive, sending SIGKILL..."
  pkill -9 -f tracking_episode_collection_camera_only 2>/dev/null || true
fi

echo "[kill_all] Done."
