#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# test_mode_gen.sh
#
# Select episodes from v3_tracking_episodes by mode (STT/DT/AT) and run
# data collection.  Output goes directly to {mode}/{scene}/{ep}/{combo}/.
#
# Usage:
#   # 3 modes, 10 episodes each (default)
#   ./test_mode_gen.sh
#
#   # Only STT, 5 episodes
#   TRACKING_MODE=stt MAX_EPISODES=5 ./test_mode_gen.sh
#
#   # Only go2_zed, 1 episode
#   TRACKING_MODE=stt MAX_EPISODES=1 ROBOT_TYPES=go2 CAMERA_TYPES=zed ./test_mode_gen.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Config ───────────────────────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
COLLECTOR="${COLLECTOR:-tracking_episode_collection_camera_only.py}"
FOLLOWER_SCRIPT="${FOLLOWER_SCRIPT:-/workspace/FLUX/datagen/follow_script/cylinder_follow_tracking.py}"
CHARACTER_SPEED="${CHARACTER_SPEED:-0.5}"
DATAGEN_NAVMESH_SNAP="${DATAGEN_NAVMESH_SNAP:-0.25}"
MAX_CONSECUTIVE_SNAP_REJECTED="${MAX_CONSECUTIVE_SNAP_REJECTED:-20}"
EARLY_ABORT_SNAP_REJECT_RATE="${EARLY_ABORT_SNAP_REJECT_RATE:-0.5}"
EARLY_ABORT_SNAP_REJECT_MAX_STEPS="${EARLY_ABORT_SNAP_REJECT_MAX_STEPS:-600}"
SAGE3D_DIR="${SAGE3D_DIR:-/mnt/ssd1/zeyingg/SAGE-3D_Official}"
ISAAC_CACHE_DIR="${ISAAC_CACHE_DIR:-$HOME/.cache/flux-isaac-sim}"
MAX_EPISODES="${MAX_EPISODES:-10}"     # max episodes per mode
TRACKING_MODE="${TRACKING_MODE:-all}"  # stt / dt / at / all
SCENE_IDS="${SCENE_IDS:-0003_839989 0004_840011}"

ROBOT_TYPES="${ROBOT_TYPES:-go2 g1 dingo}"
read -r -a ROBOTS <<< "$ROBOT_TYPES"

CAMERA_TYPES="${CAMERA_TYPES:-realsense_d435i zed}"
read -r -a CAMERAS <<< "$CAMERA_TYPES"

read -r -a GPU_ARR <<< "$GPU_IDS"

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="mode_test_${RUN_TIMESTAMP}"
HOST_RUN_DIR="$REPO_DIR/logs_formal_v3/test_runs/$RUN_NAME"
CONTAINER_RUN_DIR="/workspace/FLUX/logs_formal_v3/test_runs/$RUN_NAME"

EPISODES_V3="$SAGE3D_DIR/SAGE-3D_data/v3_tracking_episodes"  # existing dataset

# ── Validation ────────────────────────────────────────────────────────
[ -f "$REPO_DIR/sage_utils/$COLLECTOR" ] || { echo "Missing collector" >&2; exit 1; }

EXISTING="$(docker ps -a --filter "name=flux_${RUN_NAME}_" -q 2>/dev/null || true)"
[ -z "$EXISTING" ] || { echo "ERROR: containers flux_${RUN_NAME}_ already exist" >&2; exit 1; }

mkdir -p "$HOST_RUN_DIR" \
  "$ISAAC_CACHE_DIR/kit" "$ISAAC_CACHE_DIR/ov" "$ISAAC_CACHE_DIR/ov-data" \
  "$ISAAC_CACHE_DIR/pip" "$ISAAC_CACHE_DIR/glcache" "$ISAAC_CACHE_DIR/computecache"

{
  echo "run_name=$RUN_NAME"
  echo "scenes=$SCENE_IDS"
  echo "robots=${ROBOTS[*]}"
  echo "cameras=${CAMERAS[*]}"
  echo "tracking_mode=$TRACKING_MODE"
  echo "max_episodes=$MAX_EPISODES"
  echo "gpu_ids=$GPU_IDS"
  echo "max_steps=$MAX_STEPS"
  echo "save_video=$SAVE_VIDEO"
  echo "collector=$COLLECTOR"
  echo "repo_dir=$REPO_DIR"
  echo "sage3d_dir=$SAGE3D_DIR"
  echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "?")"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$HOST_RUN_DIR/run_info.txt"

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Load episode cache → select by mode
# ══════════════════════════════════════════════════════════════════════
# Pre-computed via pre_scan_scene.py → sage_utils/episode_caches/{scene_id}.json

CACHE_DIR="$REPO_DIR/sage_utils/episode_caches"

declare -A MODE_EPS  # "scene:mode" → space-separated list of original episode IDs
read -r -a ALL_MODES <<< "$(case $TRACKING_MODE in
  all) echo "stt dt at" ;; stt) echo "stt" ;; dt) echo "dt" ;; at) echo "at" ;;
esac)"

for _sid in $SCENE_IDS; do
  _cache="$CACHE_DIR/${_sid}.json"
  if [ ! -f "$_cache" ]; then
    echo "  [Phase 1] No cache for $_sid, scanning now..."
    python3 "$REPO_DIR/sage_utils/pre_scan_scene.py" "$_sid"
  fi

  # Read cache
  _stt_ids=$(python3 -c "
import json
c = json.load(open('$_cache'))
print(' '.join(str(i) for i in c['stt_ids']))
")
  _at_ids=$(python3 -c "
import json
c = json.load(open('$_cache'))
print(' '.join(str(i) for i in c['at_ids']))
")
  _dt_ids=$(python3 -c "
import json
c = json.load(open('$_cache'))
print(' '.join(str(i) for i in c['dt_ids']))
")

  IFS=' ' read -r -a _stt_arr <<< "$_stt_ids"
  IFS=' ' read -r -a _at_arr <<< "$_at_ids"
  IFS=' ' read -r -a _dt_arr <<< "$_dt_ids"

  _pick_stt=("${_stt_arr[@]:0:$MAX_EPISODES}")
  _pick_at=("${_at_arr[@]:0:$MAX_EPISODES}")
  _pick_dt=("${_dt_arr[@]:0:$MAX_EPISODES}")

  MODE_EPS["$_sid:stt"]="${_pick_stt[*]}"
  MODE_EPS["$_sid:dt"]="${_pick_dt[*]}"
  MODE_EPS["$_sid:at"]="${_pick_at[*]}"

  echo "  [Phase 1] $_sid: STT=${#_stt_arr[@]} pick=${#_pick_stt[@]}, " \
       "AT=${#_at_arr[@]} pick=${#_pick_at[@]}, DT=${#_dt_arr[@]} pick=${#_pick_dt[@]}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 1 complete — episode selection done."
echo "═══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Create temp episode dirs with patched IDs + run collector
# ══════════════════════════════════════════════════════════════════════

# For each (scene, mode, combo), run one container.
# Temp episode dir: renumbered 0..N-1, mapping saved for ori_episode_id.
_make_temp_episode_dir() {
  local _scene=$1 _mode=$2 _out=$3
  rm -rf "$_out"; mkdir -p "$_out/$_scene"
  local _orig_ids=(${MODE_EPS["$_scene:$_mode"]})
  local _new_id=0
  for _oid in "${_orig_ids[@]}"; do
    [ -z "$_oid" ] && continue
    _src="$EPISODES_V3/$_scene/episode_${_oid}.json"
    [ -f "$_src" ] || continue
    python3 -c "
import json
d = json.load(open('$_src'))
d['episode']['scene_id'] = '$_scene'
d['episode']['episode_id'] = $_new_id
d['episode']['ori_episode_id'] = $_oid
d['episode']['mode'] = '$_mode'
json.dump(d, open('$_out/$_scene/episode_${_new_id}.json', 'w'), indent=2)
" 2>/dev/null
    _new_id=$((_new_id + 1))
  done
}

ALL_COMBOS=()
for RT in "${ROBOTS[@]}"; do
  for CT in "${CAMERAS[@]}"; do
    ALL_COMBOS+=("$RT:$CT")
  done
done
TOTAL_COMBOS=${#ALL_COMBOS[@]}

# Per-GPU log files
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  [Phase 2] Collecting data"
echo "═══════════════════════════════════════════════════════════════"

_START_TS="$SECONDS"
_TOTAL_JOBS=0; _COMPLETED=0

for _sid in $SCENE_IDS; do
  [ -d "$EPISODES_V3/$_sid" ] || continue
  for _mode in "${ALL_MODES[@]}"; do
    _orig_list="${MODE_EPS["$_sid:$_mode"]}"
    [ -z "$_orig_list" ] && continue
    _n_eps=$(echo "$_orig_list" | wc -w)

    # Create temp episode dir with patched IDs
    # episode_dir basename must be the clean scene_id (USDA/semantic_map derive from it)
    _tmp_ep_dir="$HOST_RUN_DIR/tmp_episodes/${_mode}"
    _tmp_ep_container="$CONTAINER_RUN_DIR/tmp_episodes/${_mode}"
    _make_temp_episode_dir "$_sid" "$_mode" "$_tmp_ep_dir"

    _TOTAL_JOBS=$((_TOTAL_JOBS + TOTAL_COMBOS))

    for COMBO_IDX in "${!ALL_COMBOS[@]}"; do
      COMBO="${ALL_COMBOS[$COMBO_IDX]}"
      RT="${COMBO%%:*}"
      CT="${COMBO#*:}"
      _GPU="${GPU_ARR[0]}"
      _LOG_DIR="$HOST_RUN_DIR/logs/${_mode}/${_sid}/${RT}_${CT}"
      mkdir -p "$_LOG_DIR"

      _HOST_UID=$(id -u)
      _HOST_GID=$(id -g)

      if [ "$RT" = "dingo" ]; then _RR2D="0.15"; else _RR2D="0.20"; fi

      echo -n "  [${_mode}/${_sid}] ${RT}_${CT} (${_n_eps} eps) ..."

      set +e
      docker run --rm \
        --name "flux_${RUN_NAME}_${_mode}_${_sid}_${RT}_${CT}" \
        -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
        -e SAVE_VIDEO="$SAVE_VIDEO" \
        --entrypoint bash --runtime=nvidia --gpus "device=$_GPU" --network=host \
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
         -c "set +e; \
          /isaac-sim/python.sh \
          /workspace/FLUX/sage_utils/$COLLECTOR \
          --episode_dir $_tmp_ep_container \
          --scene_id $_sid \
          --robot_type $RT \
          --camera_type $CT \
          --start_idx 0 \
          --end_idx $_n_eps \
          --max_steps $MAX_STEPS \
          --character_speed $CHARACTER_SPEED \
          --robot_radius_2d $_RR2D \
          --datagen_planning_radius $_RR2D \
          --datagen_navmesh_snap $DATAGEN_NAVMESH_SNAP \
          --max_consecutive_snap_rejected $MAX_CONSECUTIVE_SNAP_REJECTED \
          --early_abort_snap_reject_rate $EARLY_ABORT_SNAP_REJECT_RATE \
          --early_abort_snap_reject_max_steps $EARLY_ABORT_SNAP_REJECT_MAX_STEPS \
          --datagen_follower_script $FOLLOWER_SCRIPT \
          --save_images \
          --image_save_dir $CONTAINER_RUN_DIR \
          --output_metrics $_LOG_DIR/metrics.csv \
           --headless; \
          echo '[chown] returning output files to host user $_HOST_UID:$_HOST_GID'; \
          chown -R $_HOST_UID:$_HOST_GID $CONTAINER_RUN_DIR || true" \
          2>&1 | tee "$_LOG_DIR/_container.log" >/dev/null
      _st=${PIPESTATUS[0]}
      set -e

      # ── Incremental render: build mp4 for this episode as soon as its
      #    png frames are written + chowned back to the host user. ──
      _EP_DIR="$HOST_RUN_DIR/${_mode}/${_sid}/0/${RT}_${CT}"
      if [ -d "$_EP_DIR/rgb" ] || [ -d "$_EP_DIR/depth" ]; then
        python3 "$REPO_DIR/sage_utils/_render_videos.py" \
          --episode_dir "$_EP_DIR" --fps 20.0 \
          >> "$_LOG_DIR/_render.log" 2>&1 \
          || echo "  WARN: video render failed for $_EP_DIR" >> "$HOST_RUN_DIR/run_info.txt"
      fi

      # ── Quarantine rejected episodes: any episode dir carrying a
      #    _REJECTED marker is moved under rejected/ so it can't be
      #    mistaken for valid training data. ──
      if [ -f "$_EP_DIR/_REJECTED" ]; then
        _REJ_DIR="$HOST_RUN_DIR/rejected/${_mode}/${_sid}/0/${RT}_${CT}"
        mkdir -p "$(dirname "$_REJ_DIR")"
        mv "$_EP_DIR" "$_REJ_DIR"
        echo "  REJECTED: $_EP_DIR -> $_REJ_DIR"
        echo "REJECTED: mode=$_mode scene=$_sid combo=${RT}_${CT} moved to rejected/" >> "$HOST_RUN_DIR/run_info.txt"
      fi

      _COMPLETED=$((_COMPLETED + 1))
      _elapsed=$((SECONDS - _START_TS))
      if [ "$_st" -eq 0 ]; then
        echo " ✓  ($((_elapsed / 60))m${_elapsed}s)"
        echo "OK: mode=$_mode scene=$_sid combo=${RT}_${CT} done" >> "$HOST_RUN_DIR/run_info.txt"
      else
        echo " ✗ exit=$_st"
        echo "WARN: mode=$_mode scene=$_sid combo=${RT}_${CT} exit_code=$_st" >> "$HOST_RUN_DIR/run_info.txt"
      fi
    done
  done
done

# ══════════════════════════════════════════════════════════════════════
# Ensure imageio-ffmpeg (required by imageio to write h264 mp4) is present
# before the per-episode incremental rendering runs.
echo ""
echo "=============================================================="
echo "  [Pre-flight] Checking video render dependencies..."
echo "=============================================================="
python3 - "$REPO_DIR" <<'PY'
import importlib.util, subprocess, sys
need = False
if importlib.util.find_spec("imageio") is None:
    need = True
else:
    try:
        import imageio
        try:
            import imageio.v2 as v2
        except ImportError:
            v2 = imageio
        v2.get_writer("/tmp/_probe.mp4", fps=1, codec="libx264")
        import os; os.remove("/tmp/_probe.mp4")
    except Exception:
        need = True
if need:
    print("  Installing imageio-ffmpeg ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
else:
    print("  imageio-ffmpeg already available")
PY

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All done. Artifacts: $HOST_RUN_DIR"
echo "═══════════════════════════════════════════════════════════════"
echo "finished_at=$(date --iso-8601=seconds)" | tee -a "$HOST_RUN_DIR/run_info.txt"
