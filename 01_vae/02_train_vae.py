import os
from typing import Dict
import numpy as np

import torch
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

from models import VanillaVAE


def train(config: Dict):
    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    dataset = datasets.ImageFolder(root='../00_assets/datasets/afhq/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    vae = VanillaVAE(in_channels=config['img_channel'], image_size=config['img_size'],
                     latent_dim=config['latent_dim']).to(device)
    print('Total trainable parameters:', sum(p.numel() for p in vae.parameters() if p.requires_grad))

    epoch_awoken = config['epoch_awoken']
    if epoch_awoken is not None:
        assert isinstance(epoch_awoken, int)
        vae.load_state_dict(torch.load(
            os.path.join(config['model_dir'], f'ckpt_{epoch_awoken}.pth'),
            map_location=device), strict=False)
        print(f'Model weight has loaded from ckpt_{epoch_awoken}.pth!')
        base = epoch_awoken
    else:
        base = 0

    optimizer = torch.optim.AdamW(vae.parameters(), lr=config['lr'], weight_decay=1e-4)

    for epoch in range(config['epoch']):
        start_time = timer()
        losses = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            result = vae(images)
            loss = vae.loss_function(*result, M_N=0.005)['loss']
            loss.backward()
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(dataloader))
        end_time = timer()

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
              f"time: {(end_time - start_time):.3f}s, ")

        if epoch >= config['epoch_save']:
            torch.save(vae.state_dict(), os.path.join(config['model_dir'], f'ckpt_{base + epoch}.pth'))


def generate(config: Dict):
    print('Images Generating...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model and evaluate
    with torch.no_grad():
        values = torch.arange(1, config['num_class'] + 1)
        labels = values.repeat_interleave(config['nrow']).to(device)

        vae = VanillaVAE(in_channels=config['img_channel'], image_size=config['img_size'],
                         latent_dim=config['latent_dim']).to(device)
        base = config['epoch_awoken'] if config['epoch_awoken'] is not None else 0

        for i in range(config['epoch_save'] + base, config['epoch'] + base):
            ckpt = torch.load(os.path.join(config['model_dir'], f'ckpt_{i}.pth'), map_location=device)
            vae.load_state_dict(ckpt)
            vae.eval()

            images = vae.sample(config['nrow'] * 5, device)

            save_image(tensor=images,
                       fp=f'../00_assets/image/vae_generated_{i}.png',
                       nrow=config['nrow'])
            print(f'animal_face_generated_{i}.png is done!')


if __name__ == '__main__':
    modelConfig = {
        'epoch': 70,
        'epoch_save': 50,
        'epoch_awoken': None,
        'batch_size': 32,
        'latent_dim': 512,
        'channel': 128,
        'channel_multy': [1, 2, 2, 2],
        'dropout': 0.15,
        'lr': 1e-5,
        'max_lr': 1e-4,
        'beta_1': 1e-4,
        'beta_T': 0.028,
        'img_channel': 3,
        'img_size': 64,
        'grad_clip': 1.,
        'train_rand': 0.01,
        'w': 1.8,  # ????
        'nrow': 8,
        'num_class': 3,
        'model_dir': '../00_assets/model_vae/'
    }

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    os.makedirs(modelConfig['model_dir'], exist_ok=True)

    train(modelConfig)
    generate(modelConfig)
