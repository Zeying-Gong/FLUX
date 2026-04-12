#!/bin/bash
# scripts/eval_socialnav_serial_single_gpu.sh

NUM_EPISODES="${1:-100}"
SPEED="${2:-0.5}"
ALGORITHM="${3:-flux}"
GPU_ID="${4:-1}"

SCENES=(
    "Full_Warehouse"
    "Hospital"
    "Jetracer"
    "Office"
    "Warehouse"
    "Warehouse_multiple_shelves"
)

NUM_SCENES=${#SCENES[@]}
BASE_PORT=14111  # flux 对应的基础端口
TASK_OFFSET=5    # socialnav 对应的任务偏移

echo "=========================================="
echo "Single-GPU Single Instance Evaluation"
echo "Task: socialnav"
echo "Algorithm: $ALGORITHM"
echo "GPU: $GPU_ID"
echo "=========================================="

mkdir -p /workspace/FLUX/logs/socialnav_singlegpu

export OMNI_KIT_ALLOW_ROOT="1"
export OMNI_KIT_ACCEPT_EULA="YES"
export OMNI_KIT_DISABLE_WATCHDOG="1"
export DISPLAY=:0
export EXP_PATH="/isaac-sim/apps"
export ISAAC_PATH="/isaac-sim"
export CUDA_VISIBLE_DEVICES="$GPU_ID"  # 使用变量

cd /workspace/FLUX || exit 1

mkdir -p /workspace/FLUX/logs/socialnav_singlegpu/${ALGORITHM}

PORT=$((BASE_PORT + TASK_OFFSET))

# 启动对应的 checkpoint server（后台运行）
echo "Starting checkpoint server on port $PORT (GPU: $GPU_ID)..."
/workspace/IsaacLab/_isaac_sim/python.sh baselines/${ALGORITHM}/${ALGORITHM}_server.py \
    --port $PORT \
    > /workspace/FLUX/logs/socialnav_singlegpu/${ALGORITHM}/server_${ALGORITHM}_port${PORT}.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待server启动并验证
echo "Waiting for server to start..."
sleep 10

# 检查server是否还在运行
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ ERROR: Server failed to start! Check logs/socialnav_singlegpu/${ALGORITHM}/server_${ALGORITHM}_port${PORT}.log"
    exit 1
fi
echo "✓ Server started successfully"

# 串行执行每个场景
for scene_idx in $(seq 0 $((NUM_SCENES - 1))); do
    scene_name="${SCENES[$scene_idx]}"
    
    # 每个场景使用独立的临时目录
    ISAAC_TEMP_DIR="/tmp/isaac_sim_scene${scene_idx}_$$"
    mkdir -p "$ISAAC_TEMP_DIR"
    
    export OMNI_USER_DATA_DIR="$ISAAC_TEMP_DIR/user_data"
    export CARB_APP_DATA_DIR="$ISAAC_TEMP_DIR/carb_data"
    
    echo "========================================"
    echo "Scene $((scene_idx + 1))/$NUM_SCENES: $scene_name"
    echo "Port: $PORT, GPU: $GPU_ID"
    echo "Temp dir: $ISAAC_TEMP_DIR"
    echo "========================================"
    
    # 运行场景
    /workspace/IsaacLab/_isaac_sim/python.sh eval_socialnav_wheeled.py \
        --scene_index $scene_idx \
        --num_episodes $NUM_EPISODES \
        --speed $SPEED \
        --port $PORT \
        --gpu_id 0 \
        > logs/socialnav_singlegpu/${ALGORITHM}/scene${scene_idx}_${scene_name}.log 2>&1
    
    exit_code=$?
    
    # 清理临时目录
    echo "Cleaning temp dir: $ISAAC_TEMP_DIR"
    rm -rf "$ISAAC_TEMP_DIR"
    
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
        echo "✓ Scene $scene_name completed"
    else
        echo "✗ Scene $scene_name failed (exit code $exit_code)"
    fi
    echo "========================================"
    
    # 场景间休息，让GPU完全释放资源
    if [ $scene_idx -lt $((NUM_SCENES - 1)) ]; then
        echo "Waiting 15 seconds for GPU cleanup..."
        sleep 15
    fi
done

# 停止 checkpoint server
echo "Stopping checkpoint server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "✓ Server stopped"

echo "=========================================="
echo "All scenes completed!"
echo "=========================================="