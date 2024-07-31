import math
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""

    def __init__(self,
                 img_size: Optional[int] = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 flatten: bool = True,
                 bias: bool = True,
                 strict_img_size: bool = True,
                 dynamic_img_pad: bool = False,
                 ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


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


class VectorEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight
        else:
            return x


class SelfAttention(nn.Module):
    """
    正常 SelfAttention
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 pre_only: bool = False, qk_norm: Optional[str] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if not pre_only:
            self.proj = nn.Linear(dim, dim)
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: Tensor):
        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, self.head_dim).movedim(2, 0)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return q, k, v

    def post_attention(self, x: Tensor) -> Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        (q, k, v) = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[float] = None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# 也是个attention
class DismantledBlock(nn.Module):
    def __init__(self, hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 pre_only: bool = False,
                 rmsnorm: bool = False,
                 scale_mod_only: bool = False,
                 swiglu: bool = False,
                 qk_norm: Optional[str] = None
                 ):
        super().__init__()

        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                  pre_only=pre_only, qk_norm=qk_norm)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden_dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(mlp_hidden_dim, hidden_size)
                )
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(hidden_size, n_mods * hidden_size, bias=True))
        self.pre_only = pre_only

    def pre_attention(self, x: Tensor, c: Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(c).chunk(6, dim=1))
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn: Tensor, x: Tensor, gate_msa: Tensor, shift_mlp: Tensor, scale_mlp: Tensor,
                       gate_mlp: Tensor):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)


class JointBlock(nn.Module):
    def __init__(self, hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 pre_only: bool = False,
                 rmsnorm: bool = False,
                 scale_mod_only: bool = False,
                 swiglu: bool = False,
                 qk_norm: Optional[str] = None):
        super().__init__()

        self.context_block = DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias,
                                             pre_only, rmsnorm, scale_mod_only, swiglu, qk_norm)
        self.x_block = DismantledBlock(hidden_size, num_heads, mlp_ratio, qkv_bias,
                                       False, rmsnorm, scale_mod_only, swiglu, qk_norm)

    @staticmethod
    def block_mixing(context, x, c, context_block, x_block):
        assert context is not None, "block_mixing called with None context"
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        x_qkv, x_intermediates = x_block.pre_attention(x, c)

        o = []
        for t in range(3):
            o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
        q, k, v = tuple(o)

        attn = attention(q, k, v, x_block.attn.num_heads)
        context_attn, x_attn = (attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:])

        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)
        else:
            context = None
        x = x_block.post_attention(x_attn, *x_intermediates)
        return context, x

    def forward(self, context, x, c):
        return self.block_mixing(context, x, c, context_block=self.context_block, x_block=self.x_block)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    def __init__(self,
                 input_size: int = 32,
                 patch_size: int = 2,
                 in_channels: int = 4,
                 depth: int = 28,
                 mlp_ratio: float = 4.0,
                 learn_sigma: bool = False,
                 adm_in_channels: Optional[int] = None,
                 context_embedder_config: Optional[Dict] = None,
                 register_length: int = 0,
                 rmsnorm: bool = False,
                 scale_mod_only: bool = False,
                 swiglu: bool = False,
                 out_channels: Optional[int] = None,
                 pos_embed_scaling_factor: Optional[float] = None,
                 pos_embed_offset: Optional[float] = None,
                 pos_embed_max_size: Optional[int] = None,
                 num_patches=None,
                 qk_norm: Optional[str] = None,
                 qkv_bias: bool = True,
                 ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        # apply magic --> this defines a head_size of 64
        hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True,
                                     strict_img_size=self.pos_embed_max_size is None)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size)

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(**context_embedder_config["params"])

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu,
                       qk_norm=qk_norm)
            for i in range(depth)]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top: top + h, left: left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: Tensor, c_mod: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, "1 ... -> b ...", b=x.shape[0]),
                                 context if context is not None else torch.Tensor([]).type_as(x)), 1)

        # context is B, L', D
        # x is B, L, D
        print(f"pooled_prompt_embeds shape: {c_mod.shape}")
        for i, block in enumerate(self.joint_blocks):
            print(f"round {i}: latent shape: {x.shape}, prompt_embeds: {context.shape}")
            context, x = block(context, x, c=c_mod)

        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None, context: Optional[Tensor] = None) -> Tensor:
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t)
        if y is not None:
            y = self.y_embedder(y)
            c = c + y
        context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context)
        x = self.unpatchify(x, hw=hw)
        return x
