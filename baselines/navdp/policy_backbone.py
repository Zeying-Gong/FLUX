"""
NavDP Backbone 网络
包含各种编码器和位置编码
"""
import torch
import torch.nn as nn
import math
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码（用于 DDPM 时间步编码）
    
    原理：使用不同频率的正弦/余弦函数编码位置信息
    公式：PE(pos, 2i) = sin(pos / 10000^(2i/d))
         PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    优点：连续、可外推、位置间距离有意义
    """
    def __init__(self, dim):
        """
        Args:
            dim: 嵌入维度（必须是偶数）
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Args:
            x: 时间步 (batch,) 或位置索引
        
        Returns:
            编码向量 (batch, dim)
        """
        device = x.device
        half_dim = self.dim // 2  # 一半用 sin，一半用 cos
        
        # 计算频率：10000^(-2i/d)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # emb: [1, 0.xxx, 0.xxx, ..., 0.0001] 递减的频率
        
        # 位置 × 频率
        emb = x[:, None] * emb[None, :]  # (batch, half_dim)
        
        # 拼接 sin 和 cos
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (batch, dim)
        return emb

class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    与固定的正弦编码不同，这个编码是通过训练学习的
    适用于序列位置编码（如轨迹点、token序列）
    """
    def __init__(self, embed_dim, max_len=5000):
        """
        Args:
            embed_dim: 嵌入维度
            max_len: 最大序列长度
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        # 创建可学习的位置嵌入表
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        """
        Args:
            x: 输入序列 (batch, seq_len, embed_dim)
        
        Returns:
            位置编码 (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 创建位置索引 [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = torch.clamp(position_ids, 0, self.max_len - 1)  # 防止越界
        
        # 扩展到 batch
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
        
        # 查表获取位置编码
        position_encoding = self.position_embedding(position_ids)  # (batch, seq_len, embed_dim)
        return position_encoding

class NavDP_RGBD_Backbone(nn.Module):
    """
    RGBD 编码器（核心的观测编码模块）
    
    架构：
    1. RGB 和 Depth 各自用 DepthAnythingV2 (ViT-Small) 提取特征
    2. 拼接 RGB + Depth 的 token
    3. 用小型 Transformer Decoder 聚合成固定长度的 memory tokens
    
    输入：
        - RGB: (batch, 8, 224, 224, 3) - 8帧历史
        - Depth: (batch, 224, 224, 1) - 当前帧
    
    输出：
        - memory_token: (batch, 128, 384) - 时序+空间记忆表示
          128 = 8帧 × 16个patch
    """
    def __init__(self,
                 image_size=224,
                 embed_size=512,        # 输出维度（实际用384，这里是历史参数）
                 memory_size=8,         # 历史帧数
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size  # 8帧
        self.image_size = image_size    # 224
        self.embed_size = embed_size
        
        # ===== ViT-Small 配置 =====
        model_configs = {'vits': {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }}
        
        # ===== RGB 编码器（冻结，只用于特征提取）=====
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.rgb_model.eval()  # 冻结模式，不更新权重
        
        # ImageNet 归一化参数
        self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            
        # ===== Depth 编码器（可训练）=====
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()  # 可训练模式
        
        # ===== Transformer 聚合器 =====
        # Query: 用于聚合大量 token 到固定长度
        self.former_query = LearnablePositionalEncoding(384, self.memory_size * 16)  # 128个query
        
        # Key/Value 的位置编码
        self.former_pe = LearnablePositionalEncoding(384, (self.memory_size + 1) * 256)
        # (8+1)*256 = 2304，为 RGB(8*256) + Depth(1*256) 留足空间
        
        # 2层 Transformer Decoder
        self.former_net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(384, 8, batch_first=True), 
            2  # 2层
        )
        
        # 投影到目标维度
        self.project_layer = nn.Linear(384, embed_size)
        
    def forward(self, images, depths):
        """
        RGBD 编码的前向传播
        
        Args:
            images: RGB 图像
                - 单帧: (batch, 224, 224, 3)
                - 多帧: (batch, 8, 224, 224, 3)
            depths: 深度图
                - 单帧: (batch, 224, 224, 1)
                - 多帧: (batch, 8, 224, 224, 1)
        
        Returns:
            memory_token: 聚合后的记忆token (batch, 128, 384)
        
        流程：
        1. RGB/Depth 分别过 ViT 提取 token
        2. 拼接并添加位置编码
        3. Transformer 聚合到固定长度
        """
        with torch.no_grad():  # RGB encoder 冻结
            # ===== 处理 RGB =====
            if len(images.shape) == 4:
                # 单帧模式（未使用）
                tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)) / \
                                     self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
            
            elif len(images.shape) == 5:
                # 多帧模式（主要使用）
                tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3)
                # (batch, 8, 224, 224, 3) -> (batch, 8, 3, 224, 224)
                
                B, T, C, H, W = tensor_images.shape
                tensor_images = tensor_images.reshape(-1, 3, self.image_size, self.image_size)
                # (batch*8, 3, 224, 224)
                
                # ImageNet 归一化
                tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1, 3, 1, 1).to(self.device)) / \
                                     self.preprocess_std.reshape(1, 3, 1, 1).to(self.device)
                
                # ViT 提取 token（中间层特征）
                image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0]
                # 输出: (batch*8, 256, 384)
                # 256 = 14×14 patch + 2 (CLS + distill token)
                
                image_token = image_token.reshape(B, T * 256, -1)
                # (batch, 8*256, 384) = (batch, 2048, 384)
            
            # ===== 处理 Depth =====
            if len(depths.shape) == 4:
                # 单帧模式
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                # Depth 是单通道，复制3次适配 ViT（需要3通道输入）
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
            
            elif len(depths.shape) == 5:
                # 多帧模式
                tensor_depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device).permute(0, 1, 4, 2, 3)
                # (batch, 8, 224, 224, 1) -> (batch, 8, 1, 224, 224)
                
                B, T, C, H, W = tensor_depths.shape
                tensor_depths = tensor_depths.reshape(-1, 1, self.image_size, self.image_size)
                # 复制3次
                tensor_depths = torch.concat([tensor_depths, tensor_depths, tensor_depths], dim=1)
                # (batch*8, 3, 224, 224)
                
                depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]
                # (batch*8, 256, 384)
                
                depth_token = depth_token.reshape(B, T * 256, -1)
                # (batch, 8*256, 384) = (batch, 2048, 384)
            
            # ===== 拼接 RGB + Depth token =====
            former_token = torch.concat((image_token, depth_token), dim=1)
            # (batch, 2048+2048, 384) = (batch, 4096, 384)
            
            former_token = former_token + self.former_pe(former_token)
            # 添加位置编码
            
            # ===== 创建 Query（固定长度）=====
            former_query = self.former_query(
                torch.zeros((image_token.shape[0], self.memory_size * 16, 384), device=self.device)
            )
            # Query: (batch, 128, 384)
            # 128 = 8帧 × 16个query/帧
            
            # ===== Transformer Decoder 聚合 =====
            memory_token = self.former_net(former_query, former_token)
            # 输入:
            #   Query: (batch, 128, 384)
            #   Key/Value: (batch, 4096, 384)
            # 输出: (batch, 128, 384)
            # 作用：把4096个token聚合成128个memory token
            
            # ===== 投影到输出维度 =====
            memory_token = self.project_layer(memory_token)
            # Linear(384 -> 512): 实际上embed_size通常设为384，所以这层基本是恒等映射
            
            return memory_token  # (batch, 128, 384)

class NavDP_ImageGoal_Backbone(nn.Module):
    """
    图像目标编码器
    
    功能：将目标图像编码为一个token向量
    
    输入：目标图像+当前图像拼接 (batch, 224, 224, 6)
    输出：图像目标编码 (batch, 384)
    
    设计：
    - 使用 DepthAnythingV2 的 ViT backbone
    - 修改输入通道为6（目标3通道+当前3通道）
    - 全局平均池化所有 patch token
    """
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        
        # ===== 创建 ViT encoder =====
        model_configs = {'vits': {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }}
        self.imagegoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.imagegoal_encoder = self.imagegoal_encoder.pretrained.float()
        
        # ===== 修改第一层卷积：3通道 -> 6通道 =====
        # 原始 ViT 接受3通道RGB，这里改为接受6通道（目标+当前）
        self.imagegoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=6,  # 目标图像3通道 + 当前图像3通道
            out_channels=self.imagegoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.imagegoal_encoder.patch_embed.proj.kernel_size,
            stride=self.imagegoal_encoder.patch_embed.proj.stride,
            padding=self.imagegoal_encoder.patch_embed.proj.padding
        )
        self.imagegoal_encoder.eval()  # 冻结
        
        self.project_layer = nn.Linear(384, embed_size)
        
    def forward(self, images):
        """
        Args:
            images: 拼接的图像 (batch, 224, 224, 6)
                    前3通道：目标图像
                    后3通道：当前图像
        
        Returns:
            image_token: 图像目标编码 (batch, 384)
        """
        with torch.no_grad():
            assert len(images.shape) == 4  # (batch, H, W, C)
            
            # ===== 调整维度顺序 =====
            tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
            # (batch, 224, 224, 6) -> (batch, 6, 224, 224)
            
            # ===== ViT 提取特征 =====
            image_token = self.imagegoal_encoder.get_intermediate_layers(tensor_images)[0]
            # 输出: (batch, 256, 384)，256个patch token
            
            # ===== 全局平均池化 =====
            image_token = image_token.mean(dim=1)
            # (batch, 256, 384) -> (batch, 384)
            # 聚合所有 patch 的信息到一个向量
            
            # ===== 投影 =====
            image_token = self.project_layer(image_token)
            # (batch, 384) -> (batch, 384/512)
            
            return image_token

class NavDP_PixelGoal_Backbone(nn.Module):
    """
    像素目标编码器
    
    功能：将像素mask+当前图像编码为一个token向量
    
    输入：像素mask+当前图像拼接 (batch, 224, 224, 4)
        - 1通道：像素mask（目标位置=1）
        - 3通道：当前RGB图像
    
    输出：像素目标编码 (batch, 384)
    
    与 ImageGoal 的区别：
    - ImageGoal: 6通道（目标RGB+当前RGB）
    - PixelGoal: 4通道（目标mask+当前RGB）
    """
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.embed_size = embed_size
        
        # ===== 创建 ViT encoder =====
        model_configs = {'vits': {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }}
        self.pixelgoal_encoder = DepthAnythingV2(**model_configs['vits'])
        self.pixelgoal_encoder = self.pixelgoal_encoder.pretrained.float()
        
        # ===== 修改第一层卷积：3通道 -> 4通道 =====
        self.pixelgoal_encoder.patch_embed.proj = nn.Conv2d(
            in_channels=4,  # 像素mask(1) + 当前RGB(3)
            out_channels=self.pixelgoal_encoder.patch_embed.proj.out_channels,
            kernel_size=self.pixelgoal_encoder.patch_embed.proj.kernel_size,
            stride=self.pixelgoal_encoder.patch_embed.proj.stride,
            padding=self.pixelgoal_encoder.patch_embed.proj.padding
        )
        self.pixelgoal_encoder.eval()  # 冻结
        
        self.project_layer = nn.Linear(384, embed_size)
        
    def forward(self, images):
        """
        Args:
            images: 拼接的图像 (batch, 224, 224, 4)
                    第1通道：像素mask
                    第2-4通道：当前RGB
        
        Returns:
            image_token: 像素目标编码 (batch, 384)
        """
        with torch.no_grad():
            assert len(images.shape) == 4  # (batch, H, W, C)
            
            # 调整维度顺序
            tensor_images = torch.as_tensor(images, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
            # (batch, 224, 224, 4) -> (batch, 4, 224, 224)
            
            # ViT 提取特征并全局池化
            image_token = self.pixelgoal_encoder.get_intermediate_layers(tensor_images)[0].mean(dim=1)
            # (batch, 256, 384) -> (batch, 384)
            
            # 投影
            image_token = self.project_layer(image_token)
            
            return image_token

if __name__ == "__main__":
    backbone = NavDP_PixelGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,4)
    print(backbone(images).shape)
    
    backbone = NavDP_ImageGoal_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(1,224,224,6)
    print(backbone(images).shape)