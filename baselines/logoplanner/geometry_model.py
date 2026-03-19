import torch
import torch.nn as nn
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2
from policy_backbone import TokenCompressor, LearnablePositionalEncoding
from Pi3.pi3.models.pi3 import Pi3
from Pi3.pi3.models.layers.camera_head import CameraHead
from Pi3.pi3.models.layers.transformer_head import TransformerDecoder, LinearPts3d

class ExtrinctHead(CameraHead):
    def __init__(self, dim=512):
        super().__init__(dim)
        output_dim = dim
        self.fc_pose = nn.Linear(output_dim, 5)
    def forward_pose(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape

        for i in range(2):
            feat = self.res_conv[i](feat) # (B*T, 264, 512)

        # feat = self.avgpool(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()) # (B*T, 512, 12, 22) -> (B*T, 512, 1, 1)
        feat = feat.view(feat.size(0), -1) # (B*T, 512)

        feat = self.more_mlps(feat)  # [B*T, 512]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out_t = self.fc_pose(feat.float())  # [B*T,5]
        return out_t


class GeometryModel(Pi3):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            context_size=12,
            device='cuda:0'
        ):
        super().__init__(pos_type, decoder_size)
        self.context_size = context_size
        self.device = device
        self.fusion_head = nn.Linear(1024+384,1024)
        self.wp_head = nn.Linear(1024+512,1024)
        
        #  World Points Decoder
        self.world_point_decoder = TransformerDecoder(
            in_dim=1024, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.world_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # Camera Head
        self.camera_head = ExtrinctHead(dim=512)
        
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()
        
        # Odometry
        self.former_query = LearnablePositionalEncoding(512,self.context_size*16)
        self.former_pe = LearnablePositionalEncoding(512,self.context_size*264)
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(512,8,batch_first=True),2)
        self.state_layer = nn.Linear(512,384)
        self.state_compressor = TokenCompressor(embed_dim=384, 
                                              num_heads=8,
                                              target_length=1)
        self.scene_layer = nn.Linear(1024,384)
        self.scene_compressor = TokenCompressor(embed_dim=384, 
                                              num_heads=8,
                                              target_length=1)
                
    def forward_depth(self, depths):
        tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
        B,T,C,H,W = tensor_depths.shape
        tensor_depths = tensor_depths.reshape(-1,1,H,W) # (B*T, 1, H, W)
        tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
        depth_prior = self.depth_model.get_intermediate_layers(tensor_depths)[0] # (B*T, 264, 384)
        return depth_prior
    
    def forward_image(self, imgs): # (B, T, H, W, 3)
        imgs = torch.as_tensor(imgs,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
        with torch.no_grad():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                imgs = (imgs - self.image_mean) / self.image_std
                B, N, _, H, W = imgs.shape
                imgs = imgs.reshape(B*N, _, H, W)
                hidden = self.encoder(imgs, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
        return hidden
    
    def forward_state(self, camera_hidden, B, N):
        camera_hidden = camera_hidden.reshape(B, -1, 512)
        former_token = camera_hidden + self.former_pe(camera_hidden)
        former_query = self.former_query(torch.zeros((camera_hidden.shape[0], self.context_size * 16, 512),device=self.device)) # (B, T*16, 512)
        state_token = self.former_net(former_query,former_token) # (B, T*16, 512)
        state_token = self.state_layer(state_token).reshape(B*N, 16, -1) # (B*T, 16, 384)
        state_token = self.state_compressor(state_token).reshape(B, N, 384) # (B, T, 384)
        return state_token
    
    def forward_scene(self, world_point_hidden, B, N):
        scene_token = self.scene_layer(world_point_hidden) # (BN, 256, 384)
        scene_token = self.scene_compressor(scene_token).reshape(B, N, 384)
        return scene_token
    
    def forward(self, imgs, depths):
        B, N, H, W, _ = imgs.shape
        assert N == self.context_size
        patch_h, patch_w = H // 14, W // 14
        image_hidden = self.forward_image(imgs)
        depth_hidden = self.forward_depth(depths)
        metric_hidden = torch.cat([image_hidden, depth_hidden], dim=-1)
        metric_hidden = self.fusion_head(metric_hidden)
        hidden, pos = self.decode(metric_hidden, N, H, W) # (B*T, 269, 2048)
        
        # finetune decoder
        camera_hidden = self.camera_decoder(hidden, xpos=pos) # (B*T, 269, 512)
        point_hidden = self.point_decoder(hidden, xpos=pos) # (B*T, 269, 1024)
        
        world_point_hidden = torch.cat([camera_hidden, point_hidden], dim=-1) # (B*T, 269, 1024+512)
        world_point_hidden = self.wp_head(world_point_hidden)
        world_point_hidden = self.world_point_decoder(world_point_hidden, xpos=pos)[:, self.patch_start_idx:] # (B*T, 269, 1024)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1).contiguous()
            
            # camera
            camera_hidden = camera_hidden[:, self.patch_start_idx:].float() # (B*T, 269, 512)
            camera_poses = self.camera_head.forward_pose(camera_hidden, patch_h, patch_w).reshape(B, N, 5).contiguous()
            
            # world points: unproject local points using camera poses or direct prediction
            world_points = self.world_point_head([world_point_hidden], (H, W)).reshape(B, N, H, W, -1)
            world_points = torch.sign(world_points) * (torch.expm1(torch.abs(world_points))).contiguous()
        
        state_token = self.forward_state(camera_hidden, B, N) # (B, N, D)
        scene_token = self.forward_scene(world_point_hidden, B, N) # (B, N, D)
            
        return [hidden, state_token, scene_token], [camera_poses, local_points, world_points]
    
if __name__ == '__main__':
    model = GeometryModel()
    model.eval()
    model.cuda()
    
    imgs = torch.randn(2,12,168,308,3).cuda()
    depths = torch.randn(2,12,168,308,1).cuda()
    
    with torch.no_grad():
        outputs = model(imgs, depths)