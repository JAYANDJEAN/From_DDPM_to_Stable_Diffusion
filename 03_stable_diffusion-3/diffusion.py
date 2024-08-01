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
    def __init__(self,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 flatten: bool = True):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
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
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size))
        self.pre_only = pre_only

    def pre_attention(self, x: Tensor, time_embeds: Tensor):
        # x could be latent or prompt_embeds
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.adaLN_modulation(time_embeds).chunk(6, dim=1))
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(time_embeds).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(time_embeds).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(time_embeds)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn: Tensor, x: Tensor, gate_msa: Tensor, shift_mlp: Tensor, scale_mlp: Tensor,
                       gate_mlp: Tensor):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: Tensor, time_embed: Tensor) -> Tensor:
        # x could be latent or prompt_embeds
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, time_embed)
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
        self.pre_only = pre_only
        self.prompt_block = DismantledBlock(hidden_size=hidden_size, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                            pre_only=pre_only, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only,
                                            swiglu=swiglu, qk_norm=qk_norm)
        self.latent_block = DismantledBlock(hidden_size=hidden_size, num_heads=num_heads,
                                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                            pre_only=False, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only,
                                            swiglu=swiglu, qk_norm=qk_norm)

    def forward(self, latent, time_embeds, prompt_embeds):
        # block_mixing
        assert prompt_embeds is not None, "block_mixing called with None context"
        prompt_qkv, prompt_intermediates = self.prompt_block.pre_attention(prompt_embeds, time_embeds)
        latent_qkv, latent_intermediates = self.latent_block.pre_attention(latent, time_embeds)

        # cross part
        o = []
        for t in range(3):
            o.append(torch.cat((prompt_qkv[t], latent_qkv[t]), dim=1))
        q, k, v = tuple(o)

        attn = attention(q, k, v, self.latent_block.attn.num_heads)
        prompt_attn, latent_attn = (attn[:, : prompt_qkv[0].shape[1]], attn[:, prompt_qkv[0].shape[1]:])

        if not self.prompt_block.pre_only:
            prompt_embeds = self.prompt_block.post_attention(prompt_attn, *prompt_intermediates)
        else:
            prompt_embeds = None
        latent = self.latent_block.post_attention(latent_attn, *latent_intermediates)
        return latent, prompt_embeds


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, latent: Tensor, time_embeds: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(time_embeds).chunk(2, dim=1)
        latent = modulate(self.norm_final(latent), shift, scale)
        latent = self.linear(latent)
        return latent


class MMDiT(nn.Module):
    def __init__(self,
                 patch_size: int = 2,  # 分块相关
                 in_channels: int = 16,  # 噪音的channel
                 out_channels: int = 16,  # 最后输出的channel
                 depth: int = 4,  # JointBlock 的个数
                 mlp_ratio: float = 4.0,  # mlp_hidden_dim
                 adm_in_channels: int = 2048,  # CLIP/L + CLIP/G 的维度之和
                 prompt_embeds_dim: int = 4096,  # T5 的维度
                 register_length: int = 0,  # 不懂干啥的 ????????
                 learn_sigma: bool = False,
                 rmsnorm: bool = False,
                 scale_mod_only: bool = False,
                 swiglu: bool = True,
                 qkv_bias: bool = True,
                 pos_embed_scaling_factor: Optional[float] = None,
                 pos_embed_offset: Optional[float] = None,
                 pos_embed_max_size: int = 128,
                 qk_norm: Optional[str] = None,
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
        self.latent_patch_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.pooled_prompt_embedder = VectorEmbedder(adm_in_channels, hidden_size)

        self.prompt_embedder = nn.Linear(prompt_embeds_dim, hidden_size)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size))

        num_patches = pos_embed_max_size * pos_embed_max_size
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu,
                       qk_norm=qk_norm)
            for i in range(depth)]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        h, w = hw
        # patched size
        h = h // self.patch_size
        w = w // self.patch_size
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

    def unpatchify(self, latent, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        p = self.patch_size
        if hw is None:
            h = w = int(latent.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == latent.shape[1]

        latent = latent.reshape(shape=(latent.shape[0], h, w, p, p, self.out_channels))
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        imgs = latent.reshape(shape=(latent.shape[0], self.out_channels, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, latent: Tensor, time_embeds: Tensor, prompt_embeds: Tensor) -> Tensor:
        assert prompt_embeds is not None
        if self.register_length > 0:
            prompt_embeds = torch.cat((repeat(self.register, "1 ... -> b ...", b=latent.shape[0]), prompt_embeds), 1)

        for i, block in enumerate(self.joint_blocks):
            latent, prompt_embeds = block(latent, time_embeds, prompt_embeds)
            print(f"block {i}: latent shape: {latent.shape}, "
                  f"prompt_embeds shape: {prompt_embeds.shape if prompt_embeds is not None else None}")

        latent = self.final_layer(latent, time_embeds)
        return latent

    def forward(self,
                latent: Tensor,
                time: Tensor,
                pooled_prompt_embeds: Optional[Tensor] = None,
                prompt_embeds: Optional[Tensor] = None) -> Tensor:
        # latent: (batch_size, num_channel, height, width)
        # time: (batch_size, )
        # h = w = 128
        hw = latent.shape[-2:]
        # input latent shape: torch.Size([2, 16, 128, 128])
        latent = self.latent_patch_embedder(latent) + self.cropped_pos_embed(hw)
        # after PatchEmbedding and PositionEmbedding latent shape: torch.Size([2, 4096, 1536])
        time_embeds = self.time_embedder(time)
        if pooled_prompt_embeds is not None:
            pooled_prompt = self.pooled_prompt_embedder(pooled_prompt_embeds)
            time_embeds = time_embeds + pooled_prompt
        prompt_embeds = self.prompt_embedder(prompt_embeds)

        latent = self.forward_core_with_concat(latent, time_embeds, prompt_embeds)
        latent = self.unpatchify(latent, hw=hw)
        return latent
