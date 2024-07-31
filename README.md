# From-DDPM-to-Stable-Diffusion

## DDPM

1. cifar10并没有训练成功吧，训练代码，需要EMA
1. https://github.com/hkproj/pytorch-ddpm/tree/main
2. https://github.com/Michedev/DDPMs-Pytorch/tree/master
3. https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
4. https://github.com/abarankab/DDPM/tree/main

## Stable Diffusion 1

1. inference代码，了解模型结构。
2. https://github.com/kjsman/stable-diffusion-pytorch
3. https://github.com/hkproj/pytorch-stable-diffusion

## Stable Diffusion 3

1. inference代码，代码很繁琐，但也能跑吧。没兴趣优化。放弃了。
2. https://github.com/Stability-AI/sd3-ref
3. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L131
3. https://github.com/comfyanonymous/ComfyUI


------------------------------------------------------------------------------------------
Loading tokenizers and models...
You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Skipping key 'shared.weight' in safetensors file as 'shared' does not exist in python model
Models loaded.

------------------------------------------------------------------------------------------
Start Generating...
latent input shape: torch.Size([1, 16, 128, 128])
----------------------------------------------------------------------
Step 1: Encode positive prompt...
CLIP/L prompt embeds shape: torch.Size([1, 77, 768])
CLIP/L pooled prompt embeds shape: torch.Size([1, 768])
CLIP/G prompt embeds shape: torch.Size([1, 77, 1280])
CLIP/G pooled prompt embeds shape: torch.Size([1, 1280])
T5 prompt embeds shape: torch.Size([1, 77, 4096])
T5 pooled prompt embeds is None!
CLIP/L and CLIP/G concat prompt embeds shape: torch.Size([1, 77, 2048])
prompt embeds shape: torch.Size([1, 154, 4096])
pooled prompt embeds shape: torch.Size([1, 2048])
----------------------------------------------------------------------
Step 1: Encode negative prompt...
CLIP/L prompt embeds shape: torch.Size([1, 77, 768])
CLIP/L pooled prompt embeds shape: torch.Size([1, 768])
CLIP/G prompt embeds shape: torch.Size([1, 77, 1280])
CLIP/G pooled prompt embeds shape: torch.Size([1, 1280])
T5 prompt embeds shape: torch.Size([1, 77, 4096])
T5 pooled prompt embeds is None!
CLIP/L and CLIP/G concat prompt embeds shape: torch.Size([1, 77, 2048])
prompt embeds shape: torch.Size([1, 154, 4096])
pooled prompt embeds shape: torch.Size([1, 2048])
----------------------------------------------------------------------
Step 2: Sampling Loop...
noise shape: torch.Size([1, 16, 128, 128]), should be same to latent shape
sigmas: tensor([1.0000, 0.9931, 0.9861, 0.9788, 0.9713, 0.9636, 0.9557, 0.9475, 0.9391,
        0.9305, 0.9215, 0.9123, 0.9028, 0.8930, 0.8828, 0.8723, 0.8614, 0.8501,
        0.8385, 0.8264, 0.8139, 0.8008, 0.7873, 0.7733, 0.7587, 0.7434, 0.7276,
        0.7110, 0.6938, 0.6758, 0.6569, 0.6372, 0.6165, 0.5948, 0.5720, 0.5480,
        0.5228, 0.4962, 0.4681, 0.4384, 0.4069, 0.3735, 0.3380, 0.3001, 0.2598,
        0.2166, 0.1703, 0.1205, 0.0669, 0.0089, 0.0000], device='cuda:0')
sigmas length: 51
Because run cond and uncond in a batch together, so first dim of tensor doubled.
noise_scaled latent shape: torch.Size([1, 16, 128, 128])
sample_euler
s_in: tensor([1.], device='cuda:0', dtype=torch.float16), shape: torch.Size([1])
==================================================
MMDiT Loop for Step 0...
input latent shape: torch.Size([2, 16, 128, 128])
input time shape: torch.Size([2])
input pooled_prompt_embeds shape: torch.Size([2, 2048])
input prompt_embeds shape: torch.Size([2, 154, 4096])
after PatchEmbedding and PositionEmbedding latent shape: torch.Size([2, 4096, 1536])
after TimeEmbedding time shape: torch.Size([2, 1536])
time_embedding + pooled_prompt_embedding shape: torch.Size([2, 1536])
after Liner prompt_embeds shape: torch.Size([2, 154, 1536])
Looping...
block 0: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 1: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 2: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 3: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 4: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 5: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 6: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 7: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 8: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 9: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 10: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 11: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 12: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 13: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 14: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 15: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 16: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 17: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 18: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 19: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 20: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 21: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 22: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: torch.Size([2, 154, 1536])
block 23: latent shape: torch.Size([2, 4096, 1536]), prompt_embeds shape: None
after looping latent shape: torch.Size([2, 4096, 64])
after unpatchify latent shape: torch.Size([2, 16, 128, 128])
==================================================
latent out of diffusion model: torch.Size([1, 16, 128, 128])
latent format: torch.Size([1, 16, 128, 128])
Sampling done
----------------------------------------------------------------------
Step 3: Decoding latent to image...
latent shape: torch.Size([1, 16, 128, 128])
image shape: torch.Size([1, 3, 1024, 1024])
Decoded

------------------------------------------------------------------------------------------
Done!!!!!!!!!!!


https://github.com/vdumoulin/conv_arithmetic


