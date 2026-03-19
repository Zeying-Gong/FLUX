import torch
import torch.nn as nn
import math
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)  # (seq_len,)
        position_ids = torch.clamp(position_ids, 0, self.max_len - 1)  # 限制索引范围，避免越界
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        position_encoding = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        return position_encoding

class TokenCompressor(nn.Module):
    def __init__(self, embed_dim, num_heads, target_length):
        super(TokenCompressor, self).__init__()
        self.target_length = target_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Learnable target sequence using nn.Embedding
        self.target_embedding = nn.Embedding(target_length, embed_dim)
        # Positional encoding
        self.positional_encoding = SinusoidalPosEmb(embed_dim) 
        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)
        # Multi-Head Attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, padding_mask=None):
        """
        x: (bs, N, 384) - Input sequence (variable length)
        padding_mask: (bs, N) - Padding mask for input sequence (True for padding positions)
        """
        bs, token_len, _ = x.shape
        token_pe = self.token_positional_encoding(x)
        x = x + token_pe  # (bs, N, 384)
        query = self.target_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
        query_pe = self.query_positional_encoding(query)
        query = query + query_pe 
        # Cross Attention: target is Query, x is Key and Value
        out, _ = self.cross_attention(
            query=query,       # (bs, target_length, embed_dim)
            key=x,               # (bs, N, embed_dim)
            value=x,             # (bs, N, embed_dim)
            key_padding_mask=padding_mask  # (bs, N) - Mask for padding positions
        )
        return out  # (bs, target_length, embed_dim)

class NavDP_RGBD_Backbone(nn.Module):
    def __init__(self,
                 image_size=224,
                 embed_size=512,
                 memory_size=8,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.image_size = image_size
        self.embed_size = embed_size
        model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
        self.rgb_model = DepthAnythingV2(**model_configs['vits'])
        self.rgb_model = self.rgb_model.pretrained.float()
        self.preprocess_mean = torch.tensor([0.485,0.456,0.406],dtype=torch.float32)
        self.preprocess_std = torch.tensor([0.229,0.224,0.225],dtype=torch.float32)
        self.rgb_model.eval()
        
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.eval()
        self.former_query = LearnablePositionalEncoding(384,self.memory_size)
        self.former_pe = LearnablePositionalEncoding(384,(self.memory_size+1)*264) 
        self.former_net = nn.TransformerDecoder(nn.TransformerDecoderLayer(384,8,batch_first=True),2)
        self.project_layer = nn.Linear(384,embed_size)
        
    def forward(self,images,depths):
        B,T,H,W,C = images.shape
        tensor_images = torch.as_tensor(images,dtype=torch.float32,device=self.device).permute(0,1,4,2,3)
        B,T,C,H,W = tensor_images.shape
        tensor_images = tensor_images.reshape(-1,3,H,W)
        tensor_norm_images = (tensor_images - self.preprocess_mean.reshape(1,3,1,1).to(self.device))/self.preprocess_std.reshape(1,3,1,1).to(self.device)
        image_token = self.rgb_model.get_intermediate_layers(tensor_norm_images)[0].reshape(B,T*264,-1)

        tensor_depths = torch.as_tensor(depths,dtype=torch.float32,device=self.device).permute(0,3,1,2)
        tensor_depths = tensor_depths.reshape(-1,1,H,W)
        tensor_depths = torch.concat([tensor_depths,tensor_depths,tensor_depths],dim=1)
        depth_token = self.depth_model.get_intermediate_layers(tensor_depths)[0]

        former_token = torch.concat((image_token,depth_token),dim=1) + self.former_pe(torch.concat((image_token,depth_token),dim=1))
        former_query = self.former_query(torch.zeros((image_token.shape[0], self.memory_size, 384),device=self.device))
        memory_token = self.former_net(former_query,former_token)
        memory_token = self.project_layer(memory_token)
        return memory_token


if __name__ == "__main__":
    backbone = NavDP_RGBD_Backbone()
    backbone = backbone.to("cuda:0")
    images = torch.rand(4,8,168,308,3).to("cuda:0")
    depths = torch.rand(4,1,168,308,1).to("cuda:0")
    print(backbone(images,depths).shape)
