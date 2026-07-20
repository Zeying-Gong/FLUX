#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
COLLECTOR="${COLLECTOR:-tracking_episode_collection_camera_only.py}"
FOLLOWER_SCRIPT="${FOLLOWER_SCRIPT:-/workspace/FLUX/datagen/follow_script/cylinder_follow_tracking.py}"
CHARACTER_SPEED="${CHARACTER_SPEED:-0.5}"
RENDER_WARMUP_FRAMES="${RENDER_WARMUP_FRAMES:-180}"
N_EPISODES="${N_EPISODES:-3}"
DATAGEN_NAVMESH_SNAP="${DATAGEN_NAVMESH_SNAP:-0.25}"
MAX_CONSECUTIVE_SNAP_REJECTED="${MAX_CONSECUTIVE_SNAP_REJECTED:-20}"
EARLY_ABORT_SNAP_REJECT_RATE="${EARLY_ABORT_SNAP_REJECT_RATE:-0.5}"
EARLY_ABORT_SNAP_REJECT_MAX_STEPS="${EARLY_ABORT_SNAP_REJECT_MAX_STEPS:-600}"
SAGE3D_DIR="${SAGE3D_DIR:-/mnt/ssd1/zeyingg/SAGE-3D_Official}"
ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/.cache/flux-isaac-sim}"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="six_combo_test_${RUN_TIMESTAMP}"
HOST_RUN_DIR="$REPO_DIR/logs_formal_v3/test_runs/$RUN_NAME"
CONTAINER_RUN_DIR="/workspace/FLUX/logs_formal_v3/test_runs/$RUN_NAME"

SCENE_IDS="${SCENE_IDS:-0001_839920 0002_839955 0039_839888 0040_839882}"
read -r -a SCENES <<< "$SCENE_IDS"

ROBOT_TYPES="${ROBOT_TYPES:-go2 g1 dingo}"
read -r -a ROBOTS <<< "$ROBOT_TYPES"

CAMERA_TYPES="${CAMERA_TYPES:-realsense_d435i zed}"
read -r -a CAMERAS <<< "$CAMERA_TYPES"

[ -f "$REPO_DIR/sage_utils/$COLLECTOR" ] || {
  echo "Missing collector: sage_utils/$COLLECTOR" >&2; exit 1
}

EXISTING="$(docker ps -a --filter "name=flux_${RUN_NAME}_" -q 2>/dev/null || true)"
if [ -n "$EXISTING" ]; then
  echo "ERROR: containers already running/matching flux_${RUN_NAME}_ exist:" >&2
  echo "$EXISTING" >&2
  echo "Kill them with: docker rm -f $EXISTING" >&2
  exit 1
fi

mkdir -p "$HOST_RUN_DIR" \
  "$ISAAC_CACHE_DIR/kit" "$ISAAC_CACHE_DIR/ov" "$ISAAC_CACHE_DIR/ov-data" \
  "$ISAAC_CACHE_DIR/pip" "$ISAAC_CACHE_DIR/glcache" "$ISAAC_CACHE_DIR/computecache"

{
  echo "run_name=$RUN_NAME"
  echo "scenes=${SCENES[*]}"
  echo "robots=${ROBOTS[*]}"
  echo "cameras=${CAMERAS[*]}"
  echo "n_episodes=$N_EPISODES"
  echo "gpu_id=$GPU_ID"
  echo "max_steps=$MAX_STEPS"
  echo "save_video=$SAVE_VIDEO"
  echo "collector=$COLLECTOR"
  echo "follower_script=$FOLLOWER_SCRIPT"
  echo "character_speed=$CHARACTER_SPEED"
  echo "datagen_navmesh_snap=$DATAGEN_NAVMESH_SNAP"
  echo "max_consecutive_snap_rejected=$MAX_CONSECUTIVE_SNAP_REJECTED"
  echo "early_abort_snap_reject_rate=$EARLY_ABORT_SNAP_REJECT_RATE"
  echo "early_abort_snap_reject_max_steps=$EARLY_ABORT_SNAP_REJECT_MAX_STEPS"
  echo "repo_dir=$REPO_DIR"
  echo "sage3d_dir=$SAGE3D_DIR"
  echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$HOST_RUN_DIR/run_info.txt"

TOTAL=$(( ${#SCENES[@]} * ${#ROBOTS[@]} * ${#CAMERAS[@]} ))
INDEX=0

for ROBOT_TYPE in "${ROBOTS[@]}"; do
  for CAMERA_TYPE in "${CAMERAS[@]}"; do
    if [ -z "${ROBOT_RADIUS_2D:-}" ]; then
      [ "$ROBOT_TYPE" = "dingo" ] && ROBOT_RADIUS_2D="0.15" || ROBOT_RADIUS_2D="0.20"
    fi
    DATAGEN_PLANNING_RADIUS="${DATAGEN_PLANNING_RADIUS:-$ROBOT_RADIUS_2D}"

    COMBO_DIR="$HOST_RUN_DIR/${ROBOT_TYPE}_${CAMERA_TYPE}"
    CONTAINER_COMBO_DIR="$CONTAINER_RUN_DIR/${ROBOT_TYPE}_${CAMERA_TYPE}"
    mkdir -p "$COMBO_DIR"

    for SCENE_ID in "${SCENES[@]}"; do
      INDEX=$((INDEX + 1))
      EPISODE_DIR="$SAGE3D_DIR/SAGE-3D_data/v3_tracking_episodes/$SCENE_ID"
      [ -d "$EPISODE_DIR" ] || {
        echo "WARN: Missing episode directory: $EPISODE_DIR, skipping" | tee -a "$HOST_RUN_DIR/run_info.txt"
        continue
      }

      LOG_FILE="$COMBO_DIR/${SCENE_ID}.log"
      echo "[$INDEX/$TOTAL] --- robot=$ROBOT_TYPE camera=$CAMERA_TYPE scene=$SCENE_ID ($N_EPISODES episodes) ---"

      set +e
      docker run --rm \
        --name "flux_${RUN_NAME}_${ROBOT_TYPE}_${CAMERA_TYPE}_${SCENE_ID}" \
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
          --end_idx $N_EPISODES \
          --max_steps $MAX_STEPS \
          --character_speed $CHARACTER_SPEED \
          --robot_radius_2d $ROBOT_RADIUS_2D \
          --datagen_planning_radius $DATAGEN_PLANNING_RADIUS \
          --datagen_navmesh_snap $DATAGEN_NAVMESH_SNAP \
          --max_consecutive_snap_rejected $MAX_CONSECUTIVE_SNAP_REJECTED \
          --early_abort_snap_reject_rate $EARLY_ABORT_SNAP_REJECT_RATE \
          --early_abort_snap_reject_max_steps $EARLY_ABORT_SNAP_REJECT_MAX_STEPS \
          --datagen_follower_script $FOLLOWER_SCRIPT \
          --save_images \
          --image_save_dir $CONTAINER_COMBO_DIR/$SCENE_ID \
          --output_metrics $CONTAINER_COMBO_DIR/$SCENE_ID/metrics.csv \
          --headless" 2>&1 | tee "$LOG_FILE"
      DOCKER_STATUS=${PIPESTATUS[0]}
      set -e

      if [ $DOCKER_STATUS -ne 0 ]; then
        echo "WARN: robot=$ROBOT_TYPE camera=$CAMERA_TYPE scene=$SCENE_ID exited with code $DOCKER_STATUS" | tee -a "$HOST_RUN_DIR/run_info.txt"
      else
        echo "OK: robot=$ROBOT_TYPE camera=$CAMERA_TYPE scene=$SCENE_ID done" | tee -a "$HOST_RUN_DIR/run_info.txt"
      fi
    done
  done
done

echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$HOST_RUN_DIR/run_info.txt"
echo "All done. Artifacts: $HOST_RUN_DIR"
