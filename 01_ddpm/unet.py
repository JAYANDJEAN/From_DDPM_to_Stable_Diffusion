from utils import *
from torch import nn, Tensor
from typing import Optional, List
import math


def find_max_num_groups(in_channels: int) -> int:
    """
    Find the maximum number of groups for group normalization based on the number of input channels.
    Args:
        in_channels (int): Number of input channels.
    Returns:
        int: Maximum number of groups for group normalization.
    """
    for i in range(4, 0, -1):
        if in_channels % i == 0:
            return i


class PositionalEncoding(nn.Module):
    r"""PositionalEncoding
    Args:
        n_steps: the max length of sequence
        dim: the number of expected features in the encoder/decoder inputs (default=512).
    """

    def __init__(self, n_steps, dim):
        super().__init__()
        self.dim = dim
        pos_embedding = torch.zeros(n_steps, dim)

        position = torch.arange(0, n_steps).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, dim, 2) * (math.log(10000.0) / dim))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        return self.pos_embedding[x]


class Attention(nn.Module):
    """
    Attention module using Multi-head Attention mechanism.
    """

    def __init__(self, num_channels: int, num_heads: int = 1):
        """
        Initialize Attention module.
        Args:
            num_channels (int): Number of input channels.
            num_heads (int): Number of attention heads (default is 1).
        """
        super().__init__()
        self.channels = num_channels
        self.heads = num_heads
        self.attn_layer = nn.MultiheadAttention(num_channels, num_heads=num_heads)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the attention module.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after attention mechanism.
        """
        b, c, w, h = x.shape
        x = x.reshape(b, w * h, c)
        attn_output, attn_output_weights = self.attn_layer(x, x, x)
        return attn_output.reshape(b, c, w, h)


class Block(nn.Module):
    """
    Residual block with convolutional layers and additional embeddings.
    """

    def __init__(self, in_channel: int,
                 out_channel: int,
                 p_dropout: float,
                 time_emb_dim: int
                 ):
        """
        Initialize Residual block.
        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            p_dropout (float): Dropout probability.
            time_emb_dim (int): Size of the time embedding.
        """
        super(Block, self).__init__()

        num_groups_in = find_max_num_groups(in_channel)
        num_groups_out = find_max_num_groups(out_channel)

        self.conv_layer = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channel),
            nn.GELU(),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channel),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

        self.time_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channel)
        )

        self.label_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channel)
        )

        self.skip_connection = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x: Tensor, time_embed: Tensor, label_embed: Optional[Tensor] = None) -> Tensor:
        h = self.conv_layer(x)
        time_embed = self.time_layer(time_embed)
        time_embed = time_embed.view(time_embed.shape[0], time_embed.shape[1], 1, 1)
        if label_embed is None:
            h = h + time_embed
        else:
            label_embed = self.label_layer(label_embed)
            label_embed = label_embed.view(label_embed.shape[0], label_embed.shape[1], 1, 1)
            h = h + time_embed + label_embed
        return self.out_layer(h) + self.skip_connection(x)


class UNet(nn.Module):
    def __init__(self, channels: List[int],
                 time_emb_dim: int,
                 num_class: int,
                 n_steps: int = 1000,
                 dropout: float = 0.0):
        """
        Initialize UNet model.
        Args:
            channels (List[int]): List of channel sizes for each layer.
            dropout (List[float]): List of dropout probabilities for each layer.
            time_emb_dim (int): Size of the time embedding.
            num_class (int): Number of classes.
        """

        super().__init__()

        assert len(channels) >= 4

        self.channels = channels
        self.num_class = num_class
        self.atten_index = 3

        pos_emb = PositionalEncoding(n_steps, time_emb_dim)
        self.time_embedding = nn.Sequential(
            pos_emb,
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.label_embedding = nn.Sequential(
            nn.Linear(num_class, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.down_blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1], dropout, time_emb_dim)
            for i in range(len(channels) - 1)
        ])

        self.middle_block = Block(channels[-1], channels[-1], dropout, time_emb_dim)

        self.up_blocks = nn.ModuleList([
            Block((2 if i != 0 else 1) * channels[-i - 1], channels[-i - 2], dropout, time_emb_dim)
            for i in range(len(channels) - 1)
        ])
        self.dropout = nn.Dropout2d(dropout)
        self.self_attn = Attention(channels[self.atten_index])

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the UNet model.
        Args:
            x (Tensor): Input tensor. x: (BATCH, CHANNELS, HEIGHT, WIDTH)
            t (Tensor): Time tensor. t: (BATCH, )
            y (Tensor): Class tensor. y: (BATCH, )
        Returns:
            Tensor: Output tensor after processing through the UNet model.
        """
        assert x.shape[1] == self.channels[0]

        time_embed = self.time_embedding(t)
        if y is None:
            label_embed = None
        else:
            label_embed = self.label_embedding(
                nn.functional.one_hot(y, num_classes=self.num_class).float()
            )

        hs = []
        h = x

        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h, time_embed, label_embed)
            if i == self.atten_index - 1:
                h = self.self_attn(h)
            h = self.dropout(h)
            if i != (len(self.down_blocks) - 1):
                hs.append(h)

        h = self.middle_block(h, time_embed, label_embed)

        for i, up_block in enumerate(self.up_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = up_block(h, time_embed, label_embed)
            if i != (len(self.up_blocks) - 1):
                h = nn.functional.interpolate(h, size=hs[-i - 1].shape[-2:], mode='nearest')
        return h
