import os
from typing import Dict
import torch
from timeit import default_timer as timer
from torchvision.utils import save_image
from models import VanillaVAE, VQVAE
from utils import animal_faces_loader, denormalize


def train(config: Dict):
    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')
    dataloader = animal_faces_loader(config['batch_size'], config['img_size'])
    vqvae = VQVAE(in_channels=config['img_channel'], img_size=config['img_size'],
                  embedding_dim=config['embedding_dim'],
                  num_embeddings=config['num_embeddings'],
                  hidden_dims=config['hidden_dims']).to(device)
    print('Total trainable parameters:', sum(p.numel() for p in vqvae.parameters() if p.requires_grad))

    epoch_awoken = config['epoch_awoken']
    if epoch_awoken is not None:
        assert isinstance(epoch_awoken, int)
        vqvae.load_state_dict(torch.load(
            os.path.join(config['model_dir'], f'ckpt_{epoch_awoken}.pth'),
            map_location=device), strict=False)
        print(f'Model weight has loaded from ckpt_{epoch_awoken}.pth!')

    optimizer = torch.optim.AdamW(vqvae.parameters(), lr=config['lr'], weight_decay=1e-4)

    for epoch in range(config['epoch']):
        start_time = timer()
        losses = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            result = vqvae(images)
            loss = vqvae.loss_function(*result, M_N=0.005)['loss']
            loss.backward()
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(dataloader))
        end_time = timer()

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
              f"time: {(end_time - start_time):.3f}s, ")

        _, batch_image = next(enumerate(dataloader))
        images = batch_image[0].to(device)
        latent = vqvae.encode(images)
        reconstruction = vqvae.decode(latent[0])
        save_image(tensor=denormalize(images),
                   fp=f'../00_assets/image/vae_raw_{epoch}.png',
                   nrow=config['nrow'])
        save_image(tensor=denormalize(reconstruction),
                   fp=f'../00_assets/image/vae_reconstruction_{epoch}.png',
                   nrow=config['nrow'])
        print(f'animal_face_reconstruction_{epoch}.png is done!')


if __name__ == '__main__':
    modelConfig = {
        'epoch': 200,
        'epoch_save': 25,
        'epoch_awoken': None,
        'batch_size': 16,
        'embedding_dim': 4,
        'num_embeddings': 128,
        'lr': 1e-5,
        'img_channel': 3,
        'img_size': 512,
        'hidden_dims': [16, 32, 64],
        'nrow': 4,
        'model_dir': '../00_assets/model_vae/'
    }

    os.makedirs(modelConfig['model_dir'], exist_ok=True)

    train(modelConfig)
