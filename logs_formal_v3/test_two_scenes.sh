#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
COLLECTOR="${COLLECTOR:-tracking_episode_collection_datagen.py}"
ROBOT_TYPE="${ROBOT_TYPE:-go2}"
CAMERA_TYPE="${CAMERA_TYPE:-zed}"
CHARACTER_SPEED="${CHARACTER_SPEED:-0.5}"
RENDER_WARMUP_FRAMES="${RENDER_WARMUP_FRAMES:-180}"
DATAGEN_PLANNING_RADIUS="${DATAGEN_PLANNING_RADIUS:-0.20}"
DATAGEN_NAVMESH_SNAP="${DATAGEN_NAVMESH_SNAP:-0.25}"
MAX_CONSECUTIVE_SNAP_REJECTED="${MAX_CONSECUTIVE_SNAP_REJECTED:-20}"
SAGE3D_DIR="${SAGE3D_DIR:-/mnt/ssd1/zeyingg/SAGE-3D_Official}"
ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/.cache/flux-isaac-sim}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="two_scenes_test_${ROBOT_TYPE}_${CAMERA_TYPE}_${RUN_TIMESTAMP}"
HOST_RUN_DIR="$REPO_DIR/logs_formal_v3/test_runs/$RUN_NAME"
CONTAINER_RUN_DIR="/workspace/FLUX/logs_formal_v3/test_runs/$RUN_NAME"

SCENES=("0001_839920" "0039_839888")

[ -f "$REPO_DIR/sage_utils/$COLLECTOR" ] || {
  echo "Missing collector: sage_utils/$COLLECTOR" >&2; exit 1
}

mkdir -p "$HOST_RUN_DIR" \
  "$ISAAC_CACHE_DIR/kit" "$ISAAC_CACHE_DIR/ov" "$ISAAC_CACHE_DIR/ov-data" \
  "$ISAAC_CACHE_DIR/pip" "$ISAAC_CACHE_DIR/glcache" "$ISAAC_CACHE_DIR/computecache"

{
  echo "run_name=$RUN_NAME"
  echo "scenes=${SCENES[*]}"
  echo "gpu_id=$GPU_ID"
  echo "max_steps=$MAX_STEPS"
  echo "save_video=$SAVE_VIDEO"
  echo "collector=$COLLECTOR"
  echo "robot_type=$ROBOT_TYPE"
  echo "camera_type=$CAMERA_TYPE"
  echo "character_speed=$CHARACTER_SPEED"
  echo "datagen_planning_radius=$DATAGEN_PLANNING_RADIUS"
  echo "datagen_navmesh_snap=$DATAGEN_NAVMESH_SNAP"
  echo "max_consecutive_snap_rejected=$MAX_CONSECUTIVE_SNAP_REJECTED"
  echo "repo_dir=$REPO_DIR"
  echo "sage3d_dir=$SAGE3D_DIR"
  echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$HOST_RUN_DIR/run_info.txt"

for SCENE_ID in "${SCENES[@]}"; do
  EPISODE_DIR="$SAGE3D_DIR/SAGE-3D_data/v3_tracking_episodes/$SCENE_ID"
  [ -d "$EPISODE_DIR" ] || {
    echo "WARN: Missing episode directory: $EPISODE_DIR, skipping" >&2
    continue
  }

  LOG_FILE="$HOST_RUN_DIR/${SCENE_ID}.log"
  echo "--- Processing scene $SCENE_ID (2 episodes) ---"

  set +e
  docker run --rm -i \
    --name "flux_${RUN_NAME}_${SCENE_ID}" \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -e SAVE_VIDEO="$SAVE_VIDEO" \
    -e RENDER_WARMUP_FRAMES="$RENDER_WARMUP_FRAMES" \
    --entrypoint bash --runtime=nvidia --gpus "device=$GPU_ID" --network=host \
    -v "$REPO_DIR:/workspace/FLUX" \
    -v "$REPO_DIR/sage_utils/patched_base_command.py:/isaac-sim/extscache/omni.anim.people-0.7.9+107.3.3/omni/anim/people/scripts/commands/base_command.py" \
    -v "$SAGE3D_DIR:/workspace/SAGE-3D_Official" \
    -v "$ISAAC_CACHE_DIR/kit:/isaac-sim/kit/cache" \
    -v "$ISAAC_CACHE_DIR/ov:/root/.cache/ov" \
    -v "$ISAAC_CACHE_DIR/ov-data:/root/.local/share/ov/data" \
    -v "$ISAAC_CACHE_DIR/pip:/root/.cache/pip" \
    -v "$ISAAC_CACHE_DIR/glcache:/root/.cache/nvidia/GLCache" \
    -v "$ISAAC_CACHE_DIR/computecache:/root/.nv/ComputeCache" \
    -w /workspace \
    quay.io/zeyinggong/flux:v2_deploy \
    -c "/isaac-sim/python.sh \
      /workspace/FLUX/sage_utils/$COLLECTOR \
      --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes/$SCENE_ID \
      --robot_type $ROBOT_TYPE \
      --camera_type $CAMERA_TYPE \
      --start_idx 0 \
      --end_idx 2 \
      --max_steps $MAX_STEPS \
      --character_speed $CHARACTER_SPEED \
      --datagen_planning_radius $DATAGEN_PLANNING_RADIUS \
      --datagen_navmesh_snap $DATAGEN_NAVMESH_SNAP \
      --max_consecutive_snap_rejected $MAX_CONSECUTIVE_SNAP_REJECTED \
      --save_images \
      --image_save_dir $CONTAINER_RUN_DIR/$SCENE_ID \
      --output_metrics $CONTAINER_RUN_DIR/$SCENE_ID/metrics.csv \
      --headless" 2>&1 | tee "$LOG_FILE"
  DOCKER_STATUS=${PIPESTATUS[0]}
  set -e

  if [ $DOCKER_STATUS -ne 0 ]; then
    echo "WARN: scene $SCENE_ID exited with code $DOCKER_STATUS" | tee -a "$HOST_RUN_DIR/run_info.txt"
  else
    echo "OK: scene $SCENE_ID done" | tee -a "$HOST_RUN_DIR/run_info.txt"
  fi
done

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$HOST_RUN_DIR/run_info.txt"
echo "All done. Artifacts: $HOST_RUN_DIR"
