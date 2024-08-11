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
    vae = VQVAE(in_channels=config['img_channel'], img_size=config['img_size'],
                embedding_dim=config['embedding_dim'], num_embeddings=config['num_embeddings']).to(device)
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
            # indices = labels == 1
            # images = images[indices]
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
        vae = VQVAE(in_channels=config['img_channel'], img_size=config['img_size'],
                    embedding_dim=config['embedding_dim'], num_embeddings=config['num_embeddings']).to(device)
        base = config['epoch_awoken'] if config['epoch_awoken'] is not None else 0

        for i in range(config['epoch_save'] + base, config['epoch'] + base):
            ckpt = torch.load(os.path.join(config['model_dir'], f'ckpt_{i}.pth'), map_location=device)
            vae.load_state_dict(ckpt)
            vae.eval()

            images = vae.sample(config['nrow'] * 5, device)

            save_image(tensor=denormalize(images),
                       fp=f'../00_assets/image/vae_generated_{i}.png',
                       nrow=config['nrow'])
            print(f'animal_face_generated_{i}.png is done!')


if __name__ == '__main__':
    modelConfig = {
        'epoch': 70,
        'epoch_save': 50,
        'epoch_awoken': None,
        'batch_size': 32,
        # 'latent_dim': 512,
        'embedding_dim': 4,
        'num_embeddings': 256,
        'lr': 1e-4,
        'img_channel': 3,
        'img_size': 512,
        'nrow': 8,
        'model_dir': '../00_assets/model_vae/'
    }

    os.makedirs(modelConfig['model_dir'], exist_ok=True)

    train(modelConfig)
    generate(modelConfig)
