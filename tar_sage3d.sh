#!/bin/bash
set -e
BASE=/mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data
DST=zeyingg@m5.precognition.team
echo "[$(date)] start usdz"
tar -C "$BASE" -cf - usdz | ssh "$DST" "mkdir -p /mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data && tar -C /mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data -xf -"
echo "[$(date)] done usdz"
echo "[$(date)] start collision"
tar -C "$BASE" -cf - collision | ssh "$DST" "tar -C /mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data -xf -"
echo "[$(date)] done collision"
echo "[$(date)] start semantic_maps_v2 usda v3_tracking_episodes"
tar -C "$BASE" -cf - semantic_maps_v2 usda v3_tracking_episodes | ssh "$DST" "tar -C /mnt/ssd1/zeyingg/SAGE-3D_Official/SAGE-3D_data -xf -"
echo "[$(date)] ALL SAGE3D DONE"
