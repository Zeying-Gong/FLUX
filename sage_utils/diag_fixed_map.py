# diag_fixed_map.py
import json, math, numpy as np
from scipy.ndimage import distance_transform_edt

sem_json = "/workspace/SAGE-3D_Official/SAGE-3D_data/temp/2D_Semantic_Map_0026_839976_Complete_fixed.json"
with open(sem_json) as f:
    data = json.load(f)

# 收集所有 mask_coords_m
all_y, all_x = [], []
for inst in data:
    for y, x in inst.get("mask_coords_m", []):
        try:
            all_y.append(float(y)); all_x.append(float(x))
        except:
            pass
if not all_y:
    print("No coordinates found!"); exit()

min_y, max_y = min(all_y), max(all_y)
min_x, max_x = min(all_x), max(all_x)
scale = 0.05
h = int(np.ceil((max_y - min_y) / scale)) + 1
w = int(np.ceil((max_x - min_x) / scale)) + 1
print(f"Map bounds: x[{min_x:.2f}, {max_x:.2f}] y[{min_y:.2f}, {max_y:.2f}]")
print(f"Grid size: {w} x {h} = {w*h} pixels")

# 构建占据网格
grid = np.zeros((h, w), dtype=np.uint8)
for inst in data:
    label = str(inst.get("category_label", "")).lower()
    if label in ("wall", "unable area"):
        for y_m, x_m in inst.get("mask_coords_m", []):
            try:
                py = int(round((float(y_m) - min_y) / scale))
                px = int(round((float(x_m) - min_x) / scale))
                if 0 <= py < h and 0 <= px < w:
                    grid[py, px] = 1
            except:
                pass

obstacle_px = (grid == 1).sum()
free_px = (grid == 0).sum()
print(f"Obstacle pixels: {obstacle_px}, Free pixels: {free_px}")
print(f"Estimated free area: {free_px * scale**2:.2f} m²")