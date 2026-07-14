#!/bin/bash
# 只跑 datagen follower 控制版 go2:zed
for GPU in 0 1 2 3; do
  S=$(( GPU * 247 ))
  E=$(( S + 247 ))
  [ $GPU -eq 3 ] && E=987
  docker run -d --name flux_go2_zed_datagen_$GPU --rm \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    --entrypoint bash --runtime=nvidia --gpus device=$GPU --network=host \
    -v /mnt/nvme1/zeyingg/FLUX:/workspace/FLUX \
    -v /mnt/ssd1/zeyingg/SAGE-3D_Official:/workspace/SAGE-3D_Official \
    -v /mnt/nvme1/zeyingg/FLUX/run_single_gpu_docker_datagen.sh:/workspace/run_single_gpu_docker_datagen.sh \
    -w /workspace quay.io/zeyinggong/flux:v2_deploy \
    -c "bash /workspace/run_single_gpu_docker_datagen.sh $GPU $S $E go2 zed"
done
