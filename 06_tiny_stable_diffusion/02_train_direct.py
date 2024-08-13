import os
from typing import Dict
import numpy as np
import torch
from timeit import default_timer as timer
from torchvision.utils import save_image
from utils import SamplerDDPM, TrainerDDPM, CosineWarmupScheduler, denormalize, animal_faces_loader
from diffusion import Diffusion
import yaml


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
            img_sample = sampler(img_noisy, values)
            save_image(tensor=denormalize(img_sample),
                       fp=f'../00_assets/image/animal_faces_direct_gen_{_epoch}.png',
                       nrow=config['nrow'],
                       padding=0)
            print(f'animal_face_generated_{_epoch}.png is done!')

    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    dataloader = animal_faces_loader(config['batch_size'], config['img_size'])
    diffusion = Diffusion(channel_img=config['img_channel'], channel_base=config['channel'],
                          num_class=config['num_class'], channel_multy=config['channel_multy'],
                          dropout=config['dropout']).to(device)
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
                                      warmup_epochs=config['epoch'] // 7,
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
            labels = labels.to(device) + 1
            if np.random.rand() < config['train_rand']:
                labels = torch.zeros_like(labels).to(device)
            loss = trainer(images, labels).sum() / bs ** 2.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), config['grad_clip'])
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(dataloader))
        end_time = timer()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
              f"time: {(end_time - start_time):.3f}s, "
              f"lr_cur: {current_lr:.7f}, lr_base: {config['lr']:.7f}")

        scheduler.step()

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(diffusion.state_dict(),
                       os.path.join(config['model_dir'], f"ckpt_{(base + config['epoch']):03}.pth"))

        generate(base + epoch)


if __name__ == '__main__':
    with open('../00_assets/yml/tiny_sd_direct.yml', 'r') as file:
        model_config = yaml.safe_load(file)

    os.makedirs(model_config['model_dir'], exist_ok=True)
    train(model_config)
