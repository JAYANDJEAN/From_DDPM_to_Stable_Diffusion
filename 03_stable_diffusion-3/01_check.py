import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from diffusion import *
from modelsummary import summary


def check_diffusion():
    diffusion = MMDiT(depth=24)
    latent = torch.randn((batch_size * 2, 16, 128, 128), dtype=torch.float32)
    # 768 + 1280 = 2048
    pooled_prompt_embeds = torch.randn((batch_size * 2, 2048), dtype=torch.float32)
    timestep = torch.randint(low=1, high=50, size=(batch_size * 2,))
    prompt_embeds = torch.randn((batch_size * 2, 154, prompt_dim), dtype=torch.float32)

    diffusion_out = diffusion(latent, timestep, pooled_prompt_embeds, prompt_embeds)
    # summary(diffusion, latent, timestep, pooled_prompt_embeds, prompt_embeds, show_input=True)


def check_parts():
    time_emb = TimestepEmbedder(hidden_size)
    self_atten = SelfAttention(hidden_size)
    mlp_hidden_dim = int(hidden_size * 4)
    mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
    patch = PatchEmbed(2, 16, hidden_size)

    time = torch.tensor(range(time_step))
    time_embedding = time_emb(time)
    assert time_embedding.shape == (time_step, hidden_size)

    x = torch.randn((batch_size, prompt_dim, hidden_size))
    # 即输入输出的shape相同
    assert self_atten(x).shape == (batch_size, prompt_dim, hidden_size)

    assert mlp(x).shape == (batch_size, prompt_dim, hidden_size)


if __name__ == '__main__':
    time_step = 50
    hidden_size = 1536  # 64 * 24
    prompt_dim = 4096
    batch_size = 1
    check_diffusion()
