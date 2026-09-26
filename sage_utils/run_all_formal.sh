#!/bin/bash
# ==============================================================
#  FLUX 正式数据采集脚本
#
#  tmux 启动:
#    tmux new-session -s flux_collect
#    bash /workspace/FLUX/sage_utils/run_all_formal.sh
#
#  脱离 tmux: Ctrl+B, D
#  重连:      tmux attach -t flux_collect
# ==============================================================

# ─── 配置 ─────────────────────────────────────────────────────
NUM_EPISODES=5          # 每场景前 N 个 episode
MAX_SCENES=100          # 场景数 (0=所有, 建议 100-200 起步)
SCENE_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"
SCRIPT="/workspace/FLUX/sage_utils/tracking_episode_collection.py"
PYTHON="/isaac-sim/python.sh"

echo "=========================================="
echo "  FLUX Formal Data Collection"
echo "  $(date)"
echo "  Episodes/scene: $NUM_EPISODES"
echo "=========================================="
echo ""

# 获取场景列表
SCENES=($(ls "$SCENE_DIR"))
[ "$MAX_SCENES" -gt 0 ] && SCENES=("${SCENES[@]:0:$MAX_SCENES}")
echo "Total scenes: ${#SCENES[@]}"

# ─── 每组合运行一个 GPU ────────────────────────────────────────
run_combo() {
  local ROBOT=$1 CAM=$2 GPU=$3
  local COMBO_NAME="${ROBOT}_${CAM}"
  echo "[GPU$GPU] Starting $COMBO_NAME (${#SCENES[@]} scenes)..."
  for SCENE in "${SCENES[@]}"; do
    echo "[GPU$GPU] $COMBO_NAME scene $SCENE..."
    CUDA_VISIBLE_DEVICES=$GPU EXP_PATH=/isaac-sim/apps $PYTHON "$SCRIPT" \
      --episode_dir "$SCENE_DIR/$SCENE" \
      --robot_type "$ROBOT" --camera_type "$CAM" \
      --max_steps 100 --character_speed 0.4 \
      --save_images --save_image_every 1 \
      --start_idx 0 --end_idx $NUM_EPISODES \
      --headless > /dev/null 2>&1
    echo "[GPU$GPU] $COMBO_NAME $SCENE done"
  done
  echo "[GPU$GPU] $COMBO_NAME complete"
}

# 6 combos, 4 GPU → 两轮
echo "Round 1: 4 combos on GPU0-3"
run_combo go2 realsense_d435i 0 &
PID0=$!
run_combo go2 zed 1 &
PID1=$!
run_combo dingo realsense_d435i 2 &
PID2=$!
run_combo dingo zed 3 &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
echo "Round 1 done"

echo "Round 2: 2 combos on GPU0-1"
run_combo g1 realsense_d435i 0 &
PID0=$!
run_combo g1 zed 1 &
PID1=$!
wait $PID0 $PID1
echo "Round 2 done"

echo ""
echo "All complete! $(date)"
echo "Output: /workspace/FLUX/logs_formal/{go2,dingo,g1}_{realsense_d435i,zed}/"
