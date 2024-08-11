import math
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

"""
基于SD1简化而来的模型
"""


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dim = dim

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        # t: (batch_size, ) -> (batch_size, hidden_size)
        t_freq = self.timestep_embedding(t, self.dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, n_channels: int):
        super().__init__()
        self.in_proj = nn.Linear(n_channels, 3 * n_channels, bias=False)
        self.out_proj = nn.Linear(n_channels, n_channels)
        self.n_heads = n_heads
        self.dim_head = n_channels // n_heads

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.dim_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(interim_shape).transpose(1, 2), (q, k, v))

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(input_shape)
        out = self.out_proj(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=False)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=False)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=True)
        self.n_heads = n_heads
        self.dim_head = d_embed // n_heads

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.dim_head)

        q, k, v = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        q, k, v = map(lambda t: t.view(interim_shape).transpose(1, 2), (q, k, v))

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(input_shape)
        out = self.out_proj(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dropout: float = 0.0, n_time=512):
        super().__init__()
        assert ch_in % 32 == 0
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.GroupNorm(32, ch_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )
        # 也为了调节 shape
        self.linear_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_time, ch_out)
        )
        if ch_in != ch_out:
            self.residual_layer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        h = self.conv_1(x)
        time_embed = self.linear_time(time_embed)
        h = h + time_embed.unsqueeze(-1).unsqueeze(-1)
        h = self.conv_2(h)
        return h + self.residual_layer(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, n_head: int = 8, d_context: int = 512):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        )
        self.atten_1 = nn.Sequential(
            nn.LayerNorm(channels),
            SelfAttention(n_head, channels)
        )
        self.norm_2 = nn.LayerNorm(channels)
        self.atten_2 = CrossAttention(n_head, channels, d_context)
        self.norm_3 = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        residue_long = x

        x = self.conv_1(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(-1, -2)  # (n, c, hw) ->(n, hw, c)

        x = self.atten_1(x) + x

        x = self.atten_2(self.norm_2(x), context) + x

        residue_short = x
        x = self.norm_3(x)
        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_2(x)
        x += residue_short

        x = x.transpose(-1, -2).view((n, c, h, w))  # (n, c, hw) -> (n, c, h, w)

        return self.conv_output(x) + residue_long


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Diffusion(nn.Module):
    def __init__(self, channel_img: int,
                 channel_multy: List[int],
                 channel_base: int = 128,
                 num_class: int = 10,
                 dropout: float = 0.0,
                 time_emb_dim: int = 512):
        super().__init__()
        d_model = 256
        assert len(channel_multy) == 4
        multy = [channel_base * i for i in channel_multy]

        self.time_embedding = TimestepEmbedder(time_emb_dim, dim=d_model)
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_class + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # width = height = 64 multy = [128, 256, 512, 512]
        self.encoders = nn.ModuleList([
            # 0 (bs, 128, 64, 64)
            SwitchSequential(nn.Conv2d(channel_img, multy[0], kernel_size=3, padding=1)),
            # 1 (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(multy[0], multy[0], dropout=dropout), AttentionBlock(multy[0])),
            # 2 (bs, 128, 32, 32)
            SwitchSequential(nn.Conv2d(multy[0], multy[0], kernel_size=3, stride=2, padding=1)),
            # 3 (bs, 256, 32, 32)
            SwitchSequential(ResidualBlock(multy[0], multy[1], dropout=dropout), AttentionBlock(multy[1])),
            # 4 (bs, 256, 16, 16)
            SwitchSequential(nn.Conv2d(multy[1], multy[1], kernel_size=3, stride=2, padding=1)),
            # 5 (bs, 512, 16, 16)
            SwitchSequential(ResidualBlock(multy[1], multy[2], dropout=dropout), AttentionBlock(multy[2])),
            # 6 (bs, 512, 8, 8)
            SwitchSequential(nn.Conv2d(multy[2], multy[2], kernel_size=3, stride=2, padding=1)),
            # 7 (bs, 512, 8, 8)
            SwitchSequential(ResidualBlock(multy[2], multy[3], dropout=dropout)),
        ])

        self.bottleneck = SwitchSequential(
            ResidualBlock(multy[3], multy[3]),
            AttentionBlock(multy[3]),
            ResidualBlock(multy[3], multy[3])
        )

        self.decoders = nn.ModuleList([
            # 8+7 = (bs, 512, 8, 8)
            SwitchSequential(ResidualBlock(multy[3] * 2, multy[2], dropout=dropout)),
            # +6 = (bs, 512, 16, 16)
            SwitchSequential(ResidualBlock(multy[2] * 2, multy[2], dropout=dropout),
                             UpSample(multy[2])),
            # +5 = (bs, 256, 16, 16)
            SwitchSequential(ResidualBlock(multy[2] * 2, multy[1], dropout=dropout),
                             AttentionBlock(multy[1])),
            # +4 = (bs, 256, 32, 32)
            SwitchSequential(ResidualBlock(multy[1] * 2, multy[1], dropout=dropout),
                             AttentionBlock(multy[1]),
                             UpSample(multy[1])),
            # +3 = (bs, 128, 32, 32)
            SwitchSequential(ResidualBlock(multy[1] * 2, multy[0], dropout=dropout),
                             AttentionBlock(multy[0])),
            # +2 = (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(multy[0] * 2, multy[0], dropout=dropout),
                             AttentionBlock(multy[0]),
                             UpSample(multy[0])),
            # +1 = (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(multy[0] * 2, multy[0], dropout=dropout),
                             AttentionBlock(multy[0])),
            # +0 = (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(multy[0] * 2, multy[0], dropout=dropout),
                             AttentionBlock(multy[0])),
        ])

        self.tail = nn.Sequential(
            nn.GroupNorm(32, multy[0]),
            nn.SiLU(),
            nn.Conv2d(multy[0], channel_img, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, time, context):
        skip_connections = []
        context = self.label_embedding(context)
        time = self.time_embedding(time)
        for i, layers in enumerate(self.encoders):
            x = layers(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)

        for i, layers in enumerate(self.decoders):
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return self.tail(x)
