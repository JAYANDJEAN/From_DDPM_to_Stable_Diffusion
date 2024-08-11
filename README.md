# From-DDPM-to-Stable-Diffusion

## Data

1. 以下实验的训练数据均使用 animal-faces，详见：https://www.kaggle.com/datasets/andrewmvd/animal-faces
    1. ![faces](00_assets/image/animal_faces.jpg)

## Stable Diffusion 1

1. 查看 Stable Diffusion 1 的 inference 代码，熟悉模型结构，其架构示意图如下：
    1. ![sd1](00_assets/image/sd1.png)
2. 参考：
    1. https://github.com/kjsman/stable-diffusion-pytorch
    2. https://github.com/hkproj/pytorch-stable-diffusion

## Stable Diffusion 3

1. 查看 Stable Diffusion 3 的 inference 代码，熟悉模型结构，其架构示意图如下：
    1. ![sd3](00_assets/image/sd3.png)
2. 参考：
    1. https://github.com/Stability-AI/sd3-ref
    2. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L131

## VAE

1. 了解各种VAE变体，用 animal-faces 训练一个 VQVAE
   1. https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
2. 参考：
   1. https://github.com/AntixK/PyTorch-VAE
   2. https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py

## Tiny Stable Diffusion

1. 尝试自己训练一个小型的Stable Diffusion，硬件只有一张16GB的RTX-4080。
2. 方法1: 将所有512\*512的图片转成64\*64，然后在图片空间直接训练，重新生成图片如下，效果虽然不是很好，但起码狗是狗猫是猫。
    1. ![](00_assets/image/animal_faces_generated_method1.png)
3. 方法2：直接用sdxl-vae将图片转换到隐藏空间，再训练。sdxl-vae的效果还不错，如下所示。训练代码基本与方法1相同，不再重跑。
   1. ![](00_assets/image/animal_faces_raw.png)
   2. ![](00_assets/image/animal_faces_sdxl_vae_latent.png)
4. 方法3：用animal-faces数据集先训练一个VAE，用VAE将图片转换到隐藏空间再训练。
5. DDPM 代码可参考：
    1. https://github.com/hkproj/pytorch-ddpm/tree/main
    2. https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
    3. https://github.com/abarankab/DDPM/tree/main

## Multimodal

1. https://www.youtube.com/watch?v=vAmKB7iPkWw
2. https://github.com/hkproj/pytorch-paligemma
3. https://huggingface.co/openbmb
4. https://openai.com/index/gpt-4v-system-card/
5. https://huggingface.co/google/paligemma-3b-pt-224