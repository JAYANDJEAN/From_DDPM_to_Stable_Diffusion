import torch
# from sd3_infer import SD3Inferencer

height = width = 1024
latent = torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

# infer = SD3Inferencer()
# latent = infer.get_empty_latent(1024, 1204)
print(latent.shape)
print(latent)