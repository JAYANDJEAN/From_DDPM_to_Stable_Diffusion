import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from diffusion import MMDiT, TimestepEmbedder
from modelsummary import summary
import math
import matplotlib.pyplot as plt


def check_pipeline():
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id,
                                                    torch_dtype=torch.float16,
                                                    token="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")
    # pipe = pipe.to("cuda")

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image.save("output.png")


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

    x = torch.randn((2, 16, 128, 128), dtype=torch.float32)
    y = torch.randn((2, 2048), dtype=torch.float32)
    t = torch.randint(low=1, high=50, size=(2,))
    context = torch.randn((2, 154, 4096), dtype=torch.float32)

    summary(diffusion, x, t, y, context, show_input=True)


def check_parts():
    t = torch.randint(low=1, high=50, size=(2,))
    t_embedder = TimestepEmbedder(512)
    print(t_embedder(t, dtype=torch.float32).shape)
    print(t_embedder(t, dtype=torch.float32)[0, 0:5])


if __name__ == '__main__':
    check_diffusion()
