# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `model.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)

import torch, math
from safetensors import safe_open
from utils import sample_euler, SDVAE, SDClipModel, SDXLClipG, T5XXLModel, SD3Tokenizer
from PIL import Image
import numpy as np
from mmdit import MMDiT
import warnings

warnings.filterwarnings("ignore")


def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model")
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(self, shift=1.0, device=None, dtype=torch.float32, file=None, prefix=""):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0]
            }
        }
        self.diffusion_model = MMDiT(pos_embed_scaling_factor=None, pos_embed_offset=None,
                                     pos_embed_max_size=pos_embed_max_size, patch_size=patch_size, in_channels=16,
                                     depth=depth, num_patches=num_patches, adm_in_channels=adm_in_channels,
                                     context_embedder_config=context_embedder_config, device=device, dtype=dtype)
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)

    def apply_model(self, x, sigma, c_crossattn=None, y=None, debug=False):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype),
                                            debug=debug).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timestep, cond, uncond, cond_scale, debug):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(torch.cat([x, x]), torch.cat([timestep, timestep]),
                                         c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
                                         y=torch.cat([cond["y"], uncond["y"]]), debug=debug)
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor([
            [-0.0645, 0.0177, 0.1052], [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360], [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364], [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283], [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700], [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220], [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207], [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259]
        ], device="cpu")
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (((latent_image + 1) / 2)
                         .clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


class ClipG:
    def __init__(self):
        CLIPG_CONFIG = {
            "hidden_act": "gelu",
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_attention_heads": 20,
            "num_hidden_layers": 32
        }
        with safe_open("../00_assets/model_sd3/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class ClipL:
    def __init__(self):
        CLIPL_CONFIG = {
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12
        }
        with safe_open("../00_assets/model_sd3/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(layer="hidden", layer_idx=-2, device="cpu", dtype=torch.float32,
                                     layer_norm_hidden_state=False, return_projected_pooled=False,
                                     textmodel_json_config=CLIPL_CONFIG)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class T5XXL:
    def __init__(self):
        T5_CONFIG = {
            "d_ff": 10240,
            "d_model": 4096,
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": 32128
        }
        with safe_open("../00_assets/model_sd3/t5xxl.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class SD3:
    def __init__(self, shift):
        with safe_open("../00_assets/model_sd3/sd3_medium.safetensors", framework="pt", device="cpu") as f:
            self.model = BaseModel(shift=shift, file=f, prefix="model.diffusion_model.", device="cpu",
                                   dtype=torch.float16).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)


class VAE:
    def __init__(self):
        with safe_open("../00_assets/model_sd3/sd3_medium.safetensors", framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=torch.float16).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float16)


#################################################################################################


class SD3Inferencer:
    def __init__(self):
        self.vae = None
        self.tokenizer = None
        self.clip_g = None
        self.clip_l = None
        self.t5xxl = None
        self.sd3 = None

    def load(self, shift):
        print("\n" + '-' * 90)
        print("Loading tokenizers and models...")
        self.tokenizer = SD3Tokenizer()
        self.clip_g = ClipG()
        self.clip_l = ClipL()
        self.t5xxl = T5XXL()
        self.sd3 = SD3(shift)
        self.vae = VAE()
        print("Models loaded.")

    def get_empty_latent(self, width, height):
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator,
                           device="cpu").to(latent.dtype)

    def get_cond(self, prompt):
        print('-' * 70)
        title = "negative" if len(prompt) == 0 else "positive"
        print(f"Step 1: Encode {title} prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        print(f"CLIP/L prompt embeds shape: {l_out.shape}")
        # CLIP/L prompt embeds shape: torch.Size([1, 77, 768])
        print(f"CLIP/L pooled prompt embeds shape: {l_pooled.shape}")
        # CLIP/L pooled prompt embeds shape: torch.Size([1, 768])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        print(f"CLIP/G prompt embeds shape: {g_out.shape}")
        # CLIP/G prompt embeds shape: torch.Size([1, 77, 1280])
        print(f"CLIP/G pooled prompt embeds shape: {g_pooled.shape}")
        # CLIP/G pooled prompt embeds shape: torch.Size([1, 1280])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        print(f"T5 prompt embeds shape: {t5_out.shape}")
        # T5 prompt embeds shape: torch.Size([1, 77, 4096])
        print(f"T5 pooled prompt embeds is None!")
        # T5 pooled prompt embeds is None!
        lg_out = torch.cat([l_out, g_out], dim=-1)
        print(f"CLIP/L and CLIP/G concat prompt embeds shape: {lg_out.shape}")
        # CLIP/L and CLIP/G concat prompt embeds shape: torch.Size([1, 77, 2048])
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        out = torch.cat([lg_out, t5_out], dim=-2)
        print(f"prompt embeds shape: {out.shape}")
        # prompt embeds shape: torch.Size([1, 154, 4096])
        pooled = torch.cat((l_pooled, g_pooled), dim=-1)
        print(f"pooled prompt embeds shape: {pooled.shape}")
        # pooled prompt embeds shape: torch.Size([1, 2048])
        return out, pooled

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        # prompt embeds and pooled prompt embeds
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(self, latent, seed, conditioning, neg_cond, steps, cfg_scale, denoise=1.0):
        print('-' * 70)
        print("Step 2: Sampling Loop...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        print(f"noise shape: {noise.shape}, should be same to latent shape")
        # noise shape: torch.Size([1, 16, 128, 128]), should be same to latent shape
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        print(f"sigmas: {sigmas}")
        print(f"sigmas length: {len(sigmas)}")
        sigmas = sigmas[int(steps * (1 - denoise)):]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {"cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale}
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(sigmas[0], noise, latent, self.max_denoise(sigmas))
        print(f"noise_scaled latent shape: {noise_scaled.shape}")
        # noise_scaled latent shape: torch.Size([1, 16, 128, 128])
        latents = sample_euler(CFGDenoiser(self.sd3.model), noise_scaled, sigmas, extra_args=extra_args)
        print(f"latent out of diffusion model: {latents[0].shape}")
        # latent out of diffusion model: torch.Size([1, 16, 128, 128])
        latents = [SD3LatentFormat().process_out(latent) for latent in latents]
        print(f"latent format: {latent[0].shape}")
        # latent format: torch.Size([1, 16, 128, 128])
        self.sd3.model = self.sd3.model.cpu()
        print("Sampling done")
        return latents

    def vae_encode(self, image) -> torch.Tensor:
        print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        print("Encoded")
        return latent

    def vae_decode(self, latent) -> Image.Image:
        print('-' * 70)
        print("Step 3: Decoding latent to image...")
        latent = latent.cuda()
        print(f"latent shape: {latent.shape}")
        # latent shape: torch.Size([1, 16, 128, 128])
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        print(f"image shape: {image.shape}")
        # image shape: torch.Size([1, 3, 1024, 1024])
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        print("Decoded")
        return out_image

    def gen_image(self, prompt, width, height, steps, cfg_scale, seed,
                  output, init_image, denoise):
        latent = self.get_empty_latent(width, height)
        print("\n" + '-' * 90)
        print("Start Generating...")
        print(f"latent input shape: {latent.shape}")
        # latent input shape: torch.Size([1, 16, 128, 128])

        if init_image:
            image_data = Image.open(init_image)
            image_data = image_data.resize((width, height), Image.LANCZOS)
            latent = self.vae_encode(image_data)
            latent = SD3LatentFormat().process_in(latent)
        conditioning = self.get_cond(prompt)
        neg_cond = self.get_cond("")
        sampled_latents = self.do_sampling(latent, seed, conditioning, neg_cond, steps, cfg_scale,
                                           denoise if init_image else 1.0)
        for i, latent in enumerate(sampled_latents):
            image = self.vae_decode(latent)
            print(i)
            image.save(f"{output}_{i:02}.png")
        print("\n" + '-' * 90)
        print("Done!!!!!!!!!!!")
