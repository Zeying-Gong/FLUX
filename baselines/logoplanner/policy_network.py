import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_backbone import *
from geometry_model import GeometryModel

class LoGoPlanner_Policy(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 context_size=12,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        
        # input encoders
        self.rgbd_encoder = NavDP_RGBD_Backbone(image_size,token_dim,memory_size=memory_size,device=device)
        self.state_encoder = GeometryModel(context_size=context_size,device=device)
        self.point_encoder = nn.Linear(3,self.token_dim)
        
        self.start_encoder = nn.Linear(3,self.token_dim)
        self.state_decoder = TokenCompressor(embed_dim=token_dim, 
                                              num_heads=heads,
                                              target_length=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)
        
        self.input_embed = nn.Linear(3,token_dim)
        self.pg_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim//2),
            nn.ReLU(),
            nn.Linear(token_dim//2, token_dim//4),
            nn.ReLU(),
            nn.Linear(token_dim//4, 3)
        )
        self.cs_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim//2),
            nn.ReLU(),
            nn.Linear(token_dim//2, token_dim//4),
            nn.ReLU(),
            nn.Linear(token_dim//4, 3)
        )
        
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, memory_size+context_size*2+4)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        
        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,
                                       prediction_type='epsilon')
        
        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask = torch.zeros((predict_size,memory_size+context_size*2+4))
        self.cond_critic_mask[:,0:4] = float('-inf')
    
    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed,unify_token):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        cond_embedding = torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed,unify_token],dim=1) + self.cond_pos_embed(torch.cat([time_embeds,goal_embed,goal_embed,goal_embed,rgbd_embed,unify_token],dim=1))
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output
    
    def predict_critic(self,predict_trajectory,rgbd_embed,unify_token):
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond_embeddings = torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed,unify_token],dim=1) + self.cond_pos_embed(torch.cat([nogoal_embed,nogoal_embed,nogoal_embed,nogoal_embed,rgbd_embed,unify_token],dim=1))
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask.to(self.device))
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output
    
    def predict_pointgoal_action(self,start_goal,memory_rgbd,context_rgbd,sample_num=16):
        with torch.no_grad():
            tensor_start_goal = torch.as_tensor(start_goal[0:1],dtype=torch.float32,device=self.device)
            startgoal_embed = self.start_encoder(tensor_start_goal).unsqueeze(1)
            rgbd_embed = self.rgbd_encoder(memory_rgbd[0:1][..., :3], memory_rgbd[0:1, -1][..., 3:4])
            [hidden, state_token, scene_token], [camera_poses, local_points, world_points] = self.state_encoder(context_rgbd[0:1][..., :3], context_rgbd[0:1][..., 3:4]) # (B, 16*T, D)
            unify_token = torch.cat([state_token, scene_token], dim=1) # (B, T*3, D)
            state_embed = self.state_decoder(torch.cat([state_token, startgoal_embed], dim=1)) # (B, 1, D)
            sub_pointgoal_pd = self.pg_pred_mlp(state_embed).squeeze(1) # (B, 3)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0)
            state_embed = torch.repeat_interleave(state_embed,sample_num,dim=0)
            unify_token = torch.repeat_interleave(unify_token,sample_num,dim=0)
            
            noisy_action = torch.randn((sample_num * start_goal.shape[0], self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),state_embed,rgbd_embed,unify_token)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample
            
            critic_values = self.predict_critic(naction,rgbd_embed,unify_token)
            critic_values = critic_values.reshape(start_goal.shape[0],sample_num)
            
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(start_goal.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(start_goal.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(start_goal.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy(), sub_pointgoal_pd.cpu().numpy()
    
if __name__ == "__main__":
    policy = LoGoPlanner_Policy()
    policy = policy.to("cuda:0")
    memory_rgbd = torch.rand(1,8,168,308,4).to("cuda:0")
    context_rgbd = torch.rand(1,12,168,308,4).to("cuda:0")
    start_goal = torch.zeros((1,3), device="cuda:0")  # Example start goal (x, y, theta)
    
    all_trajectory, critic_values, positive_trajectory, negative_trajectory, sub_pointgoal_pd = policy.predict_pointgoal_action(start_goal, memory_rgbd, context_rgbd)
    
    print("All Trajectory Shape:", all_trajectory.shape)
    print("Critic Values Shape:", critic_values.shape)
    print("Positive Trajectory Shape:", positive_trajectory.shape)
    print("Negative Trajectory Shape:", negative_trajectory.shape)