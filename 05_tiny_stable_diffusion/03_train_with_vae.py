import os
from typing import Dict
import numpy as np
import torch
from timeit import default_timer as timer
from torchvision.utils import save_image
from utils import SamplerDDPM, TrainerDDPM, CosineWarmupScheduler, denormalize, animal_faces_loader
from diffusion import Diffusion
from diffusers.models import AutoencoderKL


def train(config: Dict):
    def generate(_epoch):
        with torch.no_grad():
            values = torch.arange(1, config['num_class'] + 1)
            values = values.repeat_interleave(config['nrow']).to(device)

            diffusion.eval()
            sampler = SamplerDDPM(diffusion, config['beta_1'], config['beta_T'],
                                  config['T'], w=config['w']).to(device)
            img_noisy = torch.randn(size=[config['num_class'] * config['nrow'], config['img_channel'],
                                          config['img_size'], config['img_size']], device=device)
            latents_image = sampler(img_noisy, values)
            decoded_image = sdxl_vae.decode(latents_image).sample
            save_image(tensor=denormalize(decoded_image),
                       fp=f'../00_assets/image/animal_faces_vae_gen_{_epoch}.png',
                       nrow=config['nrow'],
                       padding=0)
            print(f'animal_face_generated_{_epoch}.png is done!')

    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    dataloader = animal_faces_loader(config['batch_size'], config['img_size'])
    diffusion = Diffusion(channel_img=4, channel_base=config['channel'],
                          channel_multy=config['channel_multy'], dropout=config['dropout']).to(device)
    sdxl_vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    print('Total trainable parameters:', sum(p.numel() for p in diffusion.parameters() if p.requires_grad))

    if config['epoch_awoken'] is not None:
        pth_files = [f for f in os.listdir(config['model_dir']) if f.endswith('.pth')]
        assert config['epoch_awoken'] in pth_files
        diffusion.load_state_dict(
            torch.load(os.path.join(config['model_dir'], config['epoch_awoken']),
                       map_location=device),
            strict=False)
        print(f"Model weight has loaded from {config['epoch_awoken']}!")
        base = int(config['epoch_awoken'].split(".")[0].split("_")[1])
    else:
        base = 0

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                      warmup_epochs=config['epoch'] // 10,
                                      max_lr=config['max_lr'],
                                      total_epochs=config['epoch'])
    trainer = TrainerDDPM(diffusion, config['beta_1'], config['beta_T'], config['T']).to(device)
    min_train_loss = float('inf')

    for epoch in range(config['epoch']):
        start_time = timer()
        losses = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            bs = images.shape[0]
            images = images.to(device)
            latents = sdxl_vae.encode(images).latent_dist.sample()
            labels = labels.to(device) + 1
            if np.random.rand() < config['train_rand']:
                labels = torch.zeros_like(labels).to(device)
            loss = trainer(latents, labels).sum() / bs ** 2.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), config['grad_clip'])
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(dataloader))
        end_time = timer()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
              f"time: {(end_time - start_time):.3f}s, "
              f"current_lr: {current_lr:.4f}, config_lr: {config['lr']:.4f}")

        # scheduler.step()

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(diffusion.state_dict(), os.path.join(config['model_dir'], f"ckpt_{base + config['epoch']}.pth"))

        generate(base + epoch)


if __name__ == '__main__':
    modelConfig = {
        'epoch': 10,
        'epoch_awoken': None,
        'batch_size': 32,
        'img_channel': 3,
        'img_size': 512,
        'num_class': 3,
        'T': 1000,
        'beta_1': 0.0015,
        'beta_T': 0.0195,
        'channel': 128,
        'channel_multy': [1, 2, 2, 2],
        'dropout': 0.1,
        'lr': 2.0e-06,
        'max_lr': 1e-4,
        'grad_clip': 1.,
        'train_rand': 0.01,
        'w': 1.8,  # ????
        'nrow': 7,
        'model_dir': '../00_assets/model_animal3_vae/'
    }

    os.makedirs(modelConfig['model_dir'], exist_ok=True)

    train(modelConfig)
