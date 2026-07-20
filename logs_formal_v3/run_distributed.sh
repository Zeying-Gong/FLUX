#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_distributed.sh
#
# Parallel data-generation driver.  Launches ONE Isaac Sim process
# per GPU (single-GPU single-process, serial over its episode slice).
# Scenes in [SCENE_START, SCENE_END) are round-robin assigned to
# the GPUs in GPU_IDS, so every GPU works on a disjoint scene subset.
#
# Each (gpu, scene) slot runs an independent `test_mode_gen.sh`
# instance in the background, with a unique RUN_SUFFIX so their
# output dirs / docker container names never collide.
#
# Usage (test):
#   SCENE_START=0 SCENE_END=2 GPU_IDS="0 1" \
#     ROBOT_TYPES=go2 CAMERA_TYPES=zed MAX_EPISODES=1 MAX_STEPS=30 \
#     bash run_distributed.sh
#
# Usage (real, machine6 = 4 GPUs, all scenes):
#   SCENE_START=0 SCENE_END=987 GPU_IDS="0 1 2 3" \
#     MAX_EPISODES=10 MAX_STEPS=300 \
#     bash run_distributed.sh
# ──────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$REPO_DIR/sage_utils/episode_caches"

# ── Cleanup on Ctrl+C ──────────────────────────────────────────
_RUN_PID=$$
cleanup() {
    echo ""
    echo "[run_distributed] Caught SIGINT — shutting down all jobs and containers..."
    kill -- -$_RUN_PID 2>/dev/null
    docker rm -f $(docker ps --filter name=flux_ -q) 2>/dev/null
    echo "[run_distributed] All stopped."
    exit 130
}
trap cleanup SIGINT SIGTERM

# ── Config (env-overridable) ──────────────────────────────────────
SCENE_START="${SCENE_START:-0}"
SCENE_END="${SCENE_END:-2}"
GPU_IDS="${GPU_IDS:-0 1}"
ROBOT_TYPES="${ROBOT_TYPES:-go2 g1 dingo}"
CAMERA_TYPES="${CAMERA_TYPES:-realsense_d435i zed}"
TRACKING_MODE="${TRACKING_MODE:-all}"
MAX_EPISODES="${MAX_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-300}"
SLOT_PER_GPU="${SLOT_PER_GPU:-1}"

# ── Build the ordered scene list from cache files ───────────────────
mapfile -t ALL_SCENES < <(ls "$CACHE_DIR"/*.json 2>/dev/null | sed 's#.*/##; s#\.json##' | sort)
TOTAL_SCENES=${#ALL_SCENES[@]}
if [ "$TOTAL_SCENES" -eq 0 ]; then
  echo "ERROR: no scene caches in $CACHE_DIR" >&2; exit 1
fi

if [ "$SCENE_START" -lt 0 ] || [ "$SCENE_START" -ge "$TOTAL_SCENES" ]; then
  echo "ERROR: SCENE_START=$SCENE_START out of [0, $TOTAL_SCENES)" >&2; exit 1
fi
END=$SCENE_END
[ "$END" -gt "$TOTAL_SCENES" ] && END=$TOTAL_SCENES
SCENES=("${ALL_SCENES[@]:$SCENE_START:$((END - SCENE_START))}")
echo "Scenes in range: ${#SCENES[@]} (global idx $SCENE_START..$((END-1)))"

# ── Round-robin assign scenes to GPUs ─────────────────────────────
read -r -a GPUS <<< "$GPU_IDS"
N_GPU=${#GPUS[@]}
declare -a GPU_SCENES=()
for ((i=0;i<N_GPU;i++)); do GPU_SCENES[$i]=""; done

idx=0
for s in "${SCENES[@]}"; do
  g=$(( idx % N_GPU ))
  GPU_SCENES[$g]+="$s "
  idx=$((idx+1))
done

LAUNCH_DIR="$SCRIPT_DIR/formal_runs/launch_logs"
mkdir -p "$LAUNCH_DIR"
RUN_LOG="$LAUNCH_DIR/run_$(date '+%Y%m%d_%H%M%S').log"
echo "GPUs: ${GPUS[*]}  (round-robin, ${#SCENES[@]} scenes)"

# master summary of the scene split, persisted to RUN_LOG
{
  echo "=== run_distributed launch $(date) ==="
  echo "SCENE_START=$SCENE_START SCENE_END=$END (of $TOTAL_SCENES)"
  echo "GPUs: ${GPUS[*]}"
  echo "ROBOT_TYPES=$ROBOT_TYPES CAMERA_TYPES=$CAMERA_TYPES"
  echo "TRACKING_MODE=$TRACKING_MODE MAX_EPISODES=$MAX_EPISODES MAX_STEPS=$MAX_STEPS"
  for ((g=0; g<N_GPU; g++)); do
    echo "  gpu${GPUS[$g]} scenes: ${GPU_SCENES[$g]}"
  done
} | tee -a "$RUN_LOG"

# ── Launch ONE serial subshell per GPU ───────────────────────────
# Each GPU runs a single subshell that processes its assigned scenes
# ONE AT A TIME (waits for each test_mode_gen.sh to finish before
# starting the next).  This keeps exactly one Isaac Sim process
# per GPU, never overloading a card.
JOBS=0
echo "Launch logs dir: $LAUNCH_DIR"
echo "This run's master log: $RUN_LOG"
for ((g=0; g<N_GPU; g++)); do
  gpu=${GPUS[$g]}
  read -r -a SLOTS <<< "${GPU_SCENES[$g]}"
  LOG="$LAUNCH_DIR/launch_gpu${gpu}.log"
  echo "  launch gpu=$gpu (serial over ${#SLOTS[@]} scenes) -> $LOG"
  (
    for scene in "${SLOTS[@]}"; do
      [ -z "$scene" ] && continue
      short_scene="${scene:0:10}"
      RUN_NAME="${short_scene}"
      echo "[gpu$gpu] $(date '+%H:%M:%S') start scene=$scene" >> "$LOG"
      SCENE_IDS="$scene" GPU_IDS="$gpu" \
      ROBOT_TYPES="$ROBOT_TYPES" CAMERA_TYPES="$CAMERA_TYPES" \
      TRACKING_MODE="$TRACKING_MODE" MAX_EPISODES="$MAX_EPISODES" \
      MAX_STEPS="$MAX_STEPS" RUN_NAME="$RUN_NAME" \
        bash "$SCRIPT_DIR/test_mode_gen.sh" >> "$LOG" 2>&1
      echo "[gpu$gpu] $(date '+%H:%M:%S') done  scene=$scene (exit=$?)" >> "$LOG"
    done
    echo "[gpu$gpu] all scenes finished." >> "$LOG"
  ) &
  JOBS=$((JOBS+1))
done

echo "Launched $JOBS GPU-serial jobs (1 Isaac process per GPU)."
echo "Tail a GPU log, e.g.:  tail -f $SCRIPT_DIR/launch_gpu${GPUS[0]}.log"

# ── Wait for all ───────────────────────────────────────────────────
wait
echo "All $JOBS jobs finished."
