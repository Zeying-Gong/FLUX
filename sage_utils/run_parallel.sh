#!/bin/bash
# Parallel tracking collection: 6 combos × 2 scenes on 4 GPUs
# Usage: bash run_parallel.sh

SCRIPT="/workspace/FLUX/sage_utils/tracking_episode_collection.py"
PYTHON="/isaac-sim/python.sh"
SCENES=("0001_839920" "0002_839955")
EPISODE_DIR="/workspace/SAGE-3D_Official/SAGE-3D_data/v3_tracking_episodes"
COMBO_DIR="/workspace/FLUX/logs/parallel_test"
mkdir -p "$COMBO_DIR"

# CUDA_VISIBLE_DEVICES + --/physics/cudaDevice to pin each instance to one GPU
run_combo() {
    local ROBOT=$1 CAM=$2 SCENE=$3 GPU=$4 LOGFILE=$5
    CUDA_VISIBLE_DEVICES=$GPU \
    EXP_PATH=/isaac-sim/apps \
    $PYTHON "$SCRIPT" \
        --episode_dir "$EPISODE_DIR/$SCENE" \
        --robot_type "$ROBOT" \
        --camera_type "$CAM" \
        --max_steps 100 \
        --character_speed 0.4 \
        --save_images \
        --start_idx 0 --end_idx 2 \
        --headless > "$LOGFILE" 2>&1
    echo "DONE: GPU=$GPU ROBOT=$ROBOT CAM=$CAM SCENE=$SCENE exit=$?"
}

START=$(date +%s)
echo "Parallel test start: $(date)"
echo ""

# ===== Round 1: scene A, combos 0-3 on GPU 0-3 =====
echo "=== Round 1: scene A (0001_839920), 4 combos ==="
run_combo go2 realsense_d435i 0001_839920 0 "$COMBO_DIR/go2_realsense_A.log" &
PID0=$!
run_combo go2 zed 0001_839920 1 "$COMBO_DIR/go2_zed_A.log" &
PID1=$!
run_combo dingo realsense_d435i 0001_839920 2 "$COMBO_DIR/dingo_realsense_A.log" &
PID2=$!
run_combo dingo zed 0001_839920 3 "$COMBO_DIR/dingo_zed_A.log" &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
echo "Round 1 done"

# ===== Round 2: scene A combos 4-5 + scene B combos 0-1 =====
echo "=== Round 2: scene A (g1) + scene B (go2) ==="
run_combo g1 realsense_d435i 0001_839920 0 "$COMBO_DIR/g1_realsense_A.log" &
PID0=$!
run_combo g1 zed 0001_839920 1 "$COMBO_DIR/g1_zed_A.log" &
PID1=$!
run_combo go2 realsense_d435i 0002_839955 2 "$COMBO_DIR/go2_realsense_B.log" &
PID2=$!
run_combo go2 zed 0002_839955 3 "$COMBO_DIR/go2_zed_B.log" &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
echo "Round 2 done"

# ===== Round 3: scene B combos 2-5 =====
echo "=== Round 3: scene B (0002_839955), remaining combos ==="
run_combo dingo realsense_d435i 0002_839955 0 "$COMBO_DIR/dingo_realsense_B.log" &
PID0=$!
run_combo dingo zed 0002_839955 1 "$COMBO_DIR/dingo_zed_B.log" &
PID1=$!
run_combo g1 realsense_d435i 0002_839955 2 "$COMBO_DIR/g1_realsense_B.log" &
PID2=$!
run_combo g1 zed 0002_839955 3 "$COMBO_DIR/g1_zed_B.log" &
PID3=$!
wait $PID0 $PID1 $PID2 $PID3
echo "Round 3 done"

END=$(date +%s)
echo ""
echo "All done: $((END-START))s"
echo "Logs: $COMBO_DIR/*.log"
