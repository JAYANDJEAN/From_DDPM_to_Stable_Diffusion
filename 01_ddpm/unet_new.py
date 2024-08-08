import math
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from torch.nn import functional as F
from typing import Optional, List


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, dim=256):
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
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
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
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, dropout=0.0, n_time=1280):
        super().__init__()
        assert ch_in % 8 == 0
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
        self.linear_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_time, ch_out)
        )
        if ch_in != ch_out:
            self.residual_layer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: Tensor, time_embed: Tensor):
        h = self.conv_1(x)
        time_embed = self.linear_time(time_embed)
        h = h + time_embed.unsqueeze(-1).unsqueeze(-1)
        h = self.conv_2(h)
        return h + self.residual_layer(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=512):
        super().__init__()
        channels = n_head * n_embd
        self.conv_1 = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        )

        self.atten_1 = nn.Sequential(
            nn.LayerNorm(channels),
            SelfAttention(n_head, channels, in_proj_bias=False)
        )

        self.norm_2 = nn.LayerNorm(channels)
        self.atten_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)

        self.norm_3 = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor, context: Tensor):
        residue_long = x

        x = self.conv_1(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))  # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.atten_1(x) + residue_short

        residue_short = x
        x = self.atten_2(self.norm_2(x), context) + residue_short

        residue_short = x
        x = self.norm_3(x)
        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))  # (n, c, h, w)

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
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


class UNet(nn.Module):
    def __init__(self, channel_img: int,
                 num_class: int = 10,
                 time_emb_dim: int = 512):
        super().__init__()
        d_model = 256
        self.time_embedding = TimestepEmbedder(time_emb_dim)
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_class + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # width = height = 64
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(channel_img, 128, kernel_size=3, padding=1)),  # 0 (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(128, 128), AttentionBlock(8, 16)),  # 1 (bs, 128, 64, 64)
            SwitchSequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),  # 2 (bs, 128, 32, 32)
            SwitchSequential(ResidualBlock(128, 256), AttentionBlock(8, 32)),  # 3 (bs, 256, 32, 32)
            SwitchSequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),  # 4 (bs, 256, 16, 16)
            SwitchSequential(ResidualBlock(256, 512), AttentionBlock(8, 64)),  # 5 (bs, 512, 16, 16)
            SwitchSequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),  # 6 (bs, 512, 8, 8)
            SwitchSequential(ResidualBlock(512, 512)),  # 7 (bs, 512, 8, 8)
        ])
        # 8 (bs, 512, 8, 8)
        self.bottleneck = SwitchSequential(ResidualBlock(512, 512), AttentionBlock(8, 64), ResidualBlock(512, 512))

        self.decoders = nn.ModuleList([
            SwitchSequential(ResidualBlock(1024, 512)),  # 8+7 = (bs, 512, 8, 8)
            SwitchSequential(ResidualBlock(1024, 512), Upsample(512)),  # +6 = (bs, 512, 16, 16)
            SwitchSequential(ResidualBlock(1024, 256), AttentionBlock(8, 32)),  # +5 = (bs, 256, 16, 16)
            SwitchSequential(ResidualBlock(512, 256), AttentionBlock(8, 32), Upsample(256)),  # +4 = (bs, 256, 32, 32)
            SwitchSequential(ResidualBlock(512, 128), AttentionBlock(8, 16)),  # +3 = (bs, 128, 32, 32)
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16), Upsample(128)),  # +2 = (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16)),  # +1 = (bs, 128, 64, 64)
            SwitchSequential(ResidualBlock(256, 128), AttentionBlock(8, 16)),  # +0 = (bs, 128, 64, 64)
        ])

        self.tail = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, channel_img, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, context, time):
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
