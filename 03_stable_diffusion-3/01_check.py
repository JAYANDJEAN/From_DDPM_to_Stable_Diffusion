import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from diffusion import *
from modelsummary import summary


def check_diffusion():
    patch_size = 2
    depth = 24
    num_patches = 36864
    pos_embed_max_size = 192
    adm_in_channels = 2048
    context_embedder_config = {'target': 'torch.nn.Linear', 'params': {'in_features': 4096, 'out_features': 1536}}
    diffusion = MMDiT(pos_embed_scaling_factor=None, pos_embed_offset=None,
                      pos_embed_max_size=pos_embed_max_size, patch_size=patch_size, in_channels=16,
                      depth=depth, num_patches=num_patches, adm_in_channels=adm_in_channels,
                      context_embedder_config=context_embedder_config)

    latent = torch.randn((batch_size * 2, 16, 128, 128), dtype=torch.float32)
    pooled_prompt_embeds = torch.randn((batch_size * 2, 2048), dtype=torch.float32)
    timestep = torch.randint(low=1, high=50, size=(batch_size * 2,))
    prompt_embeds = torch.randn((batch_size * 2, 154, 4096), dtype=torch.float32)

    diffusion_out = diffusion(latent, timestep, pooled_prompt_embeds, prompt_embeds)
    print(diffusion_out.shape)


    # summary(diffusion, latent, timestep, pooled_prompt_embeds, prompt_embeds, show_input=True)


def check_parts():
    time_emb = TimestepEmbedder(hidden_size)
    self_atten = SelfAttention(hidden_size)
    mlp_hidden_dim = int(hidden_size * 4)
    mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)

    time = torch.tensor(range(time_step))
    time_embedding = time_emb(time)
    assert time_embedding.shape == (time_step, hidden_size)

    x = torch.randn((batch_size, dim_context, hidden_size))
    # 即输入输出的shape相同
    assert self_atten(x).shape == (batch_size, dim_context, hidden_size)

    assert mlp(x).shape == (batch_size, dim_context, hidden_size)


def check_torch():
    clip = torch.randn((1, 77, 768), dtype=torch.float32)
    clip = torch.cat([clip, clip], dim=-1)  # [1,77,1536]
    t5 = torch.randn((1, 77, 4096), dtype=torch.float32)
    clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
    print(clip.shape)  # [1, 77, 4096]

    prompt_embeds = torch.cat([clip, t5], dim=-2)
    print(prompt_embeds.shape)  # [1, 154, 4096]


if __name__ == '__main__':
    time_step = 50
    hidden_size = 1536  # 64 * 24
    dim_context = 4096
    batch_size = 1
    check_diffusion()
