#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SCENE_ID="${SCENE_ID:-0001_839920}"
GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
CHARACTER_SPEED="${CHARACTER_SPEED:-0.5}"
SAGE3D_DIR="${SAGE3D_DIR:-/mnt/ssd1/zeyingg/SAGE-3D_Official}"
ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/.cache/flux-isaac-sim}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="go2_zed_datagen_${SCENE_ID}_${RUN_TIMESTAMP}"
HOST_RUN_DIR="$REPO_DIR/logs_formal_v2/test_runs/$RUN_NAME"
CONTAINER_RUN_DIR="/workspace/FLUX/logs_formal_v2/test_runs/$RUN_NAME"
EPISODE_DIR="$SAGE3D_DIR/SAGE-3D_data/v3_tracking_episodes/$SCENE_ID"

[ -f "$REPO_DIR/sage_utils/tracking_episode_collection_datagen.py" ] || {
  echo "Missing datagen collector. Pull the latest dev branch." >&2
  exit 1
}
[ -d "$EPISODE_DIR" ] || {
  echo "Missing episode directory: $EPISODE_DIR" >&2
  exit 1
}

mkdir -p \
  "$HOST_RUN_DIR" \
  "$ISAAC_CACHE_DIR/kit" \
  "$ISAAC_CACHE_DIR/ov" \
  "$ISAAC_CACHE_DIR/ov-data" \
  "$ISAAC_CACHE_DIR/pip" \
  "$ISAAC_CACHE_DIR/glcache" \
  "$ISAAC_CACHE_DIR/computecache"

LOG_FILE="$HOST_RUN_DIR/console.log"
{
  echo "run_name=$RUN_NAME"
  echo "scene_id=$SCENE_ID"
  echo "gpu_id=$GPU_ID"
  echo "max_steps=$MAX_STEPS"
  echo "save_video=$SAVE_VIDEO"
  echo "character_speed=$CHARACTER_SPEED"
  echo "repo_dir=$REPO_DIR"
  echo "sage3d_dir=$SAGE3D_DIR"
  echo "cache_dir=$ISAAC_CACHE_DIR"
  echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$HOST_RUN_DIR/run_info.txt"

set +e
docker run --rm -i \
  --name "flux_${RUN_NAME}" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e SAVE_VIDEO="$SAVE_VIDEO" \
  --entrypoint bash \
  --runtime=nvidia \
  --gpus "device=$GPU_ID" \
  --network=host \
  -v "$REPO_DIR:/workspace/FLUX" \
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
    /workspace/FLUX/sage_utils/tracking_episode_collection_datagen.py \
    --episode_dir /workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes/$SCENE_ID \
    --robot_type go2 \
    --camera_type zed \
    --start_idx 0 \
    --end_idx 1 \
    --max_steps $MAX_STEPS \
    --character_speed $CHARACTER_SPEED \
    --save_images \
    --image_save_dir $CONTAINER_RUN_DIR/$SCENE_ID \
    --headless" 2>&1 | tee "$LOG_FILE"
DOCKER_STATUS=${PIPESTATUS[0]}
set -e

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$HOST_RUN_DIR/run_info.txt"
echo "exit_code=$DOCKER_STATUS" | tee -a "$HOST_RUN_DIR/run_info.txt"
echo "Test artifacts: $HOST_RUN_DIR"
exit "$DOCKER_STATUS"
