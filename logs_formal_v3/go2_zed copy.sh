# 只跑 go2:zed
for GPU in 0 1 2 3; do
  S=$(( GPU * 247 ))
  E=$(( S + 247 ))
  [ $GPU -eq 3 ] && E=987
  docker run -d --name flux_go2_zed_$GPU --rm \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    --entrypoint bash --gpus device=$GPU --network=host \
    -v /mnt/nvme1/zeyingg/FLUX:/workspace/FLUX \
    -v /mnt/ssd1/zeyingg/SAGE-3D_Official:/workspace/SAGE-3D_Official \
    -v /home/zeyingg/run_single_gpu_docker.sh:/workspace/run_single_gpu_docker.sh \
    -w /workspace quay.io/zeyinggong/flux:v2_deploy \
    -c "bash /workspace/run_single_gpu_docker.sh $GPU $S $E go2 zed"
done


# 其他 combo 同理，替换最后的 go2 zed 为：
# - go2 realsense_d435i
# - dingo realsense_d435i
# - dingo zed
# - g1 realsense_d435i
# - g1 zed
# 日志自动命名为 collect_{robot}_{camera}_{时间戳}_gpu{N}.log，不会覆盖。