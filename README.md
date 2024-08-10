# From-DDPM-to-Stable-Diffusion

## VAE

1. https://github.com/AntixK/PyTorch-VAE

## Tiny Stable Diffusion

1. 尝试自己训练一个小型的Stable Diffusion，硬件只有一张16GB的RTX-4080。
2. 数据集使用animal-faces，详见：https://www.kaggle.com/datasets/andrewmvd/animal-faces
   1. ![faces](00_assets/image/animal_faces.jpg)
3. 方法1: 将所有512\*512的图片转成64\*64，然后在图片空间直接训练，重新生成图片如下
   1. ![](00_assets/image/animal_face_generated_method1.png)
4. 方法2：用animal-faces数据集先训练一个VAE，用VAE将图片转换到隐藏空间再训练，目前没看懂这部分代码，搁置。
   1. https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
5. 方法3：直接用sdxl-vae将图片转换到隐藏空间，再训练。
   1. ![](00_assets/image/animal_faces_raw.png)
   2. ![](00_assets/image/animal_faces_sdxl_vae_latent.png)
6. DDPM 代码可参考：
   1. https://github.com/hkproj/pytorch-ddpm/tree/main
   2. https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
   3. https://github.com/abarankab/DDPM/tree/main


## Stable Diffusion 1

1. Stable Diffusion 1 架构示意图如下：
   1. ![sd1](00_assets/image/sd1.png)
2. inference代码可参考：
   1. https://github.com/kjsman/stable-diffusion-pytorch
   2. https://github.com/hkproj/pytorch-stable-diffusion

## Stable Diffusion 3

1. Stable Diffusion 3 架构示意图如下：
   1. ![sd3](00_assets/image/sd3.png)
2. inference代码可参考：
   1. https://github.com/Stability-AI/sd3-ref
   2. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L131
