from utils import *
from torch import nn, Tensor
from typing import Optional, List


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
                 time_embed_size: int,
                 num_classes: int
                 ):
        """
        Initialize Residual block.
        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            p_dropout (float): Dropout probability.
            time_embed_size (int): Size of the time embedding.
            num_classes (int): Number of classes.
        """
        super(Block, self).__init__()

        num_groups_in = find_max_num_groups(in_channel)
        num_groups_out = find_max_num_groups(out_channel)

        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channel),
            nn.GELU(),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.GroupNorm(num_groups_out, out_channel),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

        self.linear_time = nn.Sequential(
            nn.Linear(time_embed_size, in_channel),
            nn.SiLU(),
            nn.Linear(in_channel, in_channel)
        )

        self.linear_class = nn.Sequential(
            nn.Linear(num_classes, in_channel),
            nn.SiLU(),
            nn.Linear(in_channel, in_channel)
        )

        self.skip_connection = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x: Tensor, time_embed: Tensor, class_embed: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the residual block.
        Args:
            x (Tensor): Input tensor.
            time_embed (Tensor): Time embedding tensor.
            class_embed (Tensor): Class embedding tensor.
        Returns:
            Tensor: Output tensor after processing through the residual block.
        """
        time_embed = self.linear_time(time_embed)
        time_embed = time_embed.view(time_embed.shape[0], time_embed.shape[1], 1, 1)
        if class_embed is None:
            x = x + time_embed
        else:
            class_embed = self.linear_class(class_embed)
            class_embed = class_embed.view(class_embed.shape[0], class_embed.shape[1], 1, 1)
            x = x + time_embed + class_embed
        return self.conv(x) + self.skip_connection(x)


class UNet(nn.Module):
    def __init__(self, channels: List[int],
                 p_dropouts: List[float],
                 time_embed_size: int,
                 num_classes: int,
                 use_down: bool,
                 use_attention: bool):
        """
        Initialize UNet model.
        Args:
            channels (List[int]): List of channel sizes for each layer.
            p_dropouts (List[float]): List of dropout probabilities for each layer.
            time_embed_size (int): Size of the time embedding.
            num_classes (int): Number of classes.
            use_down (bool): Whether to use down sampling or not.
            use_attention (bool): Whether to use attention mechanism or not.
        """

        super(UNet, self).__init__()
        assert len(channels) == len(p_dropouts) + 1
        assert len(channels) >= 4

        self.channels = channels
        self.use_down = use_down
        self.use_attention = use_attention

        self.down_blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1], p_dropouts[i], time_embed_size, num_classes)
            for i in range(len(channels) - 1)
        ])

        self.middle_block = Block(channels[-1], channels[-1],
                                  p_dropouts[-1], time_embed_size, num_classes)

        self.up_blocks = nn.ModuleList([
            Block((2 if i != 0 else 1) * channels[-i - 1], channels[-i - 2],
                  p_dropouts[-i - 1], time_embed_size, num_classes)
            for i in range(len(channels) - 1)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout2d(p) for p in p_dropouts])
        self.self_attn = Attention(channels[3])

    def forward(self, x: Tensor, time_embed: Tensor, class_embed: Tensor) -> Tensor:
        """
        Forward pass of the UNet model.
        Args:
            x (Tensor): Input tensor.
            time_embed (Tensor): Time tensor.
            class_embed (Tensor): Class tensor.
        Returns:
            Tensor: Output tensor after processing through the UNet model.
        """
        assert x.shape[1] == self.channels[0]

        hs = []
        h = x

        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h, time_embed, class_embed)
            # todo
            if i == 2 and self.use_attention:
                h = self.self_attn(h)

            h = self.dropouts[i](h)
            if i != (len(self.down_blocks) - 1):
                hs.append(h)

        h = self.middle_block(h, time_embed, class_embed)

        for i, up_block in enumerate(self.up_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = up_block(h, time_embed, class_embed)
            # todo
            if self.use_down and (i != (len(self.up_blocks) - 1)):
                h = nn.functional.interpolate(h, size=hs[-i - 1].shape[-2:], mode='nearest')
        return h
