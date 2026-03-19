import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import LoGoPlanner_Policy

class LoGoPlanner_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 context_size=12,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model = "./100.ckpt",
                 device='cuda:0'):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.navi_former = LoGoPlanner_Policy(image_size,memory_size,context_size,predict_size,temporal_depth,heads,token_dim,device)
        self.navi_former.load_state_dict(torch.load(navi_model,map_location=self.device),strict=False)
        self.navi_former.to(self.device)
        self.navi_former.eval()
        self.target_H, self.target_W = 168, 308
    
    def reset(self,batch_size,threshold):
        print("================ LogoPlanner Agent Reset ================")
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for i in range(batch_size)]
        self.depth_queue = [[] for i in range(batch_size)]
        self.goal_queue = [[] for i in range(batch_size)]

    def reset_env(self,i):
        self.memory_queue[i] = []
        self.depth_queue[i] = []
        self.goal_queue[i] = []
    
    def project_trajectory(self,images,n_trajectories,n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i,:,:,0:2]
            n_value = n_values[i]
            for waypoints,value in zip(n_trajectory,n_value):
                norm_value = np.clip(-value*0.1,0,1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0],3)) - 0.2
                input_points[:,0:2] = waypoints
                input_points[:,1] = -input_points[:,1]
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                for i in range(camera_x.shape[0]-1):
                    try:
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                    except:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks,axis=1)

    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(self.target_W, self.target_H))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(self.target_W, self.target_H))
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    def process_pixel(self,pixel_coords,input_images):
        return_pixels = []
        H,W,C = input_images.shape[1],input_images.shape[2],input_images.shape[3]
        prop = self.image_size/max(H,W)
        for pixel_coord,input_image in zip(pixel_coords,input_images):
            panel_image = np.zeros_like(input_image,dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10
            
            if min_x <= 0:
                panel_image[:,0:10] = 255
            elif min_y <= 0:
                panel_image[0:10,:] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:,panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0]-10:,:] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y,min_x:max_x] = 255
            
            resize_image = cv2.resize(panel_image,(-1,-1),fx=prop,fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_pixels.append(resize_image)
        return np.array(return_pixels).mean(axis=-1)
    
    def process_pointgoal(self,goals):
        clip_goals = goals.clip(-10,10)
        clip_goals[:,0] = np.clip(clip_goals[:,0],0,10)
        return clip_goals
    
    def step_nogoal(self,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_nogoal_action(input_image,input_depth)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def get_indices(self, start_choice, current_choice, context_size):
        distance = current_choice - start_choice
        if distance < context_size:
            indices = [start_choice] * (context_size - distance - 1)
            # 添加剩余的点
            indices.extend(range(start_choice, current_choice + 1))
        else:
            # 计算步长，确保能取到8个点（包括起止点）
            step = distance / (context_size - 1)
            # 生成等间隔的索引
            indices = [start_choice + int(round(i * step)) for i in range(context_size)]
            # 确保最后一个点是current_choice
            indices[-1] = current_choice
        return indices
    
    def visualize_rgb_grid(self, rgb_data, prefix):
        """
        可视化RGB数据并保存为图像
        
        Args:
            rgb_data: RGB数据，形状为 (batch_size, num_frames, height, width, 3)
            prefix: 保存图像文件的前缀
        """
        for i in range(rgb_data.shape[0]):  # 遍历每个环境
            rgb = rgb_data[i]  # (num_frames, height, width, 3)
            
            # 创建一个网格图像来显示所有帧
            num_frames, height, width, _ = rgb.shape
            grid_cols = 4  # 网格的列数
            grid_rows = (num_frames + grid_cols - 1) // grid_cols  # 计算需要的行数
            
            # 创建空白网格
            rgb_grid = np.zeros((grid_rows * height, grid_cols * width, 3), dtype=np.uint8)
            
            # 填充网格
            for j in range(num_frames):
                row = j // grid_cols
                col = j % grid_cols
                # 将图像转换为uint8格式
                frame = (rgb[j] * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 将帧放入网格
                rgb_grid[row*height:(row+1)*height, col*width:(col+1)*width] = frame
            
            # 保存图像
            cv2.imwrite(f"{prefix}_env_{i}.jpg", rgb_grid)
    
    def step_pointgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        memory_rgbds = []
        context_rgbds = []
        for i in range(len(self.memory_queue)): # envs
            self.memory_queue[i].append(process_images[i])
            self.depth_queue[i].append(process_depths[i])
            self.goal_queue[i].append(goals[i])
            memory_length = len(self.memory_queue[i])
            indices = self.get_indices(0, memory_length - 1, self.context_size)
            input_image = np.array(self.memory_queue[i])[indices]
            input_depth = np.array(self.depth_queue[i])[indices]
            context_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))

            current_length = len(self.memory_queue[i])
            start_idx = max(current_length - self.memory_size, 0)
            indices = list(range(start_idx, current_length))

            zeros_needed = self.memory_size - len(indices)

            if zeros_needed > 0:
                indices = [0] * zeros_needed + indices       

            input_image = np.array([self.memory_queue[i][j] for j in indices])
            input_depth = np.array([self.depth_queue[i][j] for j in indices])
            memory_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))
    
        memory_rgbds = np.array(memory_rgbds) # (1, 8, 224, 224, 4)
        context_rgbds = np.array(context_rgbds) # (1, 8, 224, 224, 4)
        context_rgbds[..., -1][context_rgbds[..., -1] > 5.0] = 0
        context_rgbds[..., -1][context_rgbds[..., -1] < 0.1] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] > 5.0] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] < 0.1] = 0

        start_goal = goals
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory, sub_pointgoal_pd = self.navi_former.predict_pointgoal_action(start_goal,memory_rgbds,context_rgbds)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        
        print(all_values.max(),all_values.min())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd
