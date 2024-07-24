from utils import *
from torch import nn, Tensor
from typing import Optional, List
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEncoding(nn.Module):
    def __init__(self, n_steps, dim):
        super().__init__()
        pos_embedding = torch.zeros(n_steps, dim)
        position = torch.arange(0, n_steps).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, dim, 2) * (math.log(10000.0) / dim))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        return self.pos_embedding[x]


class DownSample(nn.Module):
    def __init__(self, ch_in: int):
        super().__init__()
        self.down = nn.Conv2d(ch_in, ch_in, 3, stride=2, padding=1)

    def forward(self, x: Tensor, time_embed: Tensor, label_embed: Optional[Tensor] = None):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1),
        )

    def forward(self, x: Tensor, time_embed: Tensor, label_embed: Optional[Tensor] = None):
        x = self.up(x)
        return x


class Attention(nn.Module):
    def __init__(self, num_channel: int, num_heads: int = 1):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(num_channel, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, num_channel, height, width) -> (batch_size, num_channel, height, width)
        b, c, w, h = x.shape
        x = x.reshape(b, w * h, c)
        attn_output, attn_output_weights = self.attn_layer(x, x, x)
        return attn_output.reshape(b, c, w, h)


class ResBlock(nn.Module):
    """
    ResBlock 不改变 (height, width)，改变 channel
    DownSample UpSample 改变 (height, width)，不改变 channel
    Attention 不改变shape
    """

    def __init__(self, ch_in: int, ch_out: int, dropout: float, dim: int, use_attn: bool = False):
        super().__init__()
        assert ch_in % 8 == 0
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, ch_in),
            Swish(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, ch_out),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )

        self.linear_time = nn.Sequential(
            Swish(),
            nn.Linear(dim, ch_out)
        )
        self.linear_label = nn.Sequential(
            Swish(),
            nn.Linear(dim, ch_out)
        )

        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if use_attn:
            self.attn = Attention(ch_out)
        else:
            self.attn = nn.Identity()

    def forward(self, x: Tensor, time_embed: Tensor, label_embed: Optional[Tensor] = None) -> Tensor:
        time_embed = self.linear_time(time_embed)
        h = self.conv1(x)
        h += time_embed[:, :, None, None]
        if label_embed is not None:
            label_embed = self.linear_label(label_embed)
            h += label_embed[:, :, None, None]

        h = self.conv2(h) + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self,
                 channel_img: int,
                 channel_base: int,
                 channel_mults: List[int],
                 num_class: int = 10,
                 num_res_blocks: int = 2,
                 time_emb_dim: int = 512,
                 n_steps: int = 1000,
                 dropout: float = 0.0):

        super().__init__()

        d_model = 128
        pos_emb = PositionalEncoding(n_steps, d_model)
        self.time_embedding = nn.Sequential(
            pos_emb,
            nn.Linear(d_model, d_model),
            Swish(),
            nn.Linear(d_model, time_emb_dim)
        )
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_class + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, d_model),
            Swish(),
            nn.Linear(d_model, time_emb_dim)
        )

        self.head = nn.Conv2d(channel_img, channel_base, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [channel_base]
        ch_cur = channel_base

        for i, mult in enumerate(channel_mults):
            ch_out = channel_base * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch_cur, ch_out, dropout, time_emb_dim, use_attn=True))
                ch_cur = ch_out
                channels.append(ch_cur)
            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(ch_cur))
                channels.append(ch_cur)

        self.mid = nn.ModuleList([ResBlock(ch_cur, ch_cur, dropout, time_emb_dim, use_attn=True),
                                  ResBlock(ch_cur, ch_cur, dropout, time_emb_dim, use_attn=False)])

        for i, mult in reversed(list(enumerate(channel_mults))):
            ch_out = channel_base * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(channels.pop() + ch_cur, ch_out, dropout, time_emb_dim, use_attn=False))
                ch_cur = ch_out
            if i != 0:
                self.ups.append(UpSample(ch_cur))

        assert len(channels) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, ch_cur),
            Swish(),
            nn.Conv2d(ch_cur, channel_img, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        time_embed = self.time_embedding(t)
        label_embed = self.label_embedding(y) if y is not None else None

        h = self.head(x)
        hs = [h]

        for layer in self.downs:
            h = layer(h, time_embed, label_embed)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, time_embed, label_embed)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, time_embed, label_embed)

        h = self.tail(h)

        return h
