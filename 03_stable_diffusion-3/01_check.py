import torch
from diffusers import StableDiffusion3Pipeline

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
