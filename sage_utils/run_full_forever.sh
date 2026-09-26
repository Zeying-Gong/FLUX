#!/bin/bash
# ==============================================================
#  FLUX 全量采集：6 combos × 987 scenes，4 GPU 并行
#  每 combo 独占一张 GPU，metrics.csv 无冲突
#
#  tmux:
#    tmux new-session -s flux_collect
#    bash /workspace/FLUX/sage_utils/run_full_forever.sh
#  脱离: Ctrl+B, D  |  重连: tmux attach -t flux_collect
# ==============================================================

NUM_EPISODES=100         # 每场景 episode 数（0..99 全量）
SCENE_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"
SCRIPT="/workspace/FLUX/sage_utils/tracking_episode_collection.py"
PYTHON="/isaac-sim/python.sh"
ALL_SCENES=($(ls "$SCENE_DIR"))
TOTAL=${#ALL_SCENES[@]}
CHUNK=$(( (TOTAL + 3) / 4 ))

echo "=========================================="
echo "  FLUX Full Collection"
echo "  Scenes: $TOTAL × $NUM_EPISODES ep × 6 combos"
echo "  4 GPU parallel, ~$CHUNK scenes each"
echo "  Each GPU handles all 6 combos on its scenes"
echo "  (不同 GPU 处理不同 scene，metrics.csv 无冲突)"
echo "  Start: $(date)"
echo "=========================================="
echo ""

run_gpu() {
  local GPU=$1 S=$2 E=$3
  for combo in \
    "go2:realsense_d435i" "go2:zed" \
    "dingo:realsense_d435i" "dingo:zed" \
    "g1:realsense_d435i" "g1:zed"; do
    local robot="${combo%%:*}"
    local cam="${combo##*:}"
    echo "[GPU$GPU] Starting ${robot}_${cam} ($((E-S)) scenes)..."
    for ((i=S; i<E; i++)); do
      scene="${ALL_SCENES[$i]}"
      CUDA_VISIBLE_DEVICES=$GPU EXP_PATH=/isaac-sim/apps $PYTHON "$SCRIPT" \
        --episode_dir "$SCENE_DIR/$scene" \
        --robot_type "$robot" --camera_type "$cam" \
        --max_steps 100 --character_speed 0.4 \
        --save_images --start_idx 0 --end_idx $NUM_EPISODES \
        --headless > /dev/null 2>&1
      echo "[GPU$GPU] ${robot}_${cam} $scene ($((i-S+1))/$((E-S)))"
    done
    echo "[GPU$GPU] ${robot}_${cam} ALL DONE ($((E-S)) scenes)"
  done
}

run_gpu 0 0 $CHUNK &
run_gpu 1 $CHUNK $((CHUNK*2)) &
run_gpu 2 $((CHUNK*2)) $((CHUNK*3)) &
run_gpu 3 $((CHUNK*3)) $TOTAL &

wait
echo ""
echo "=========================================="
echo "  All done: $(date)"
echo "=========================================="
