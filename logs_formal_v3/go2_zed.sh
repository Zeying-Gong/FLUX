# 只跑 go2:zed
for GPU in 0 1 2 3; do
  S=$(( GPU * 247 ))
  E=$(( S + 247 ))
  [ $GPU -eq 3 ] && E=987
  docker run -d --name flux_go2_zed_$GPU --rm \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    --entrypoint bash --runtime=nvidia --gpus device=$GPU --network=host \
    -v /mnt/nvme1/zeyingg/FLUX:/workspace/FLUX \
    -v /mnt/ssd1/zeyingg/SAGE-3D_Official:/workspace/SAGE-3D_Official \
    -v /home/zeyingg/run_single_gpu_docker.sh:/workspace/run_single_gpu_docker.sh \
    -w /workspace quay.io/zeyinggong/flux:v2_deploy \
    -c "bash /workspace/run_single_gpu_docker.sh $GPU $S $E go2 zed"
done