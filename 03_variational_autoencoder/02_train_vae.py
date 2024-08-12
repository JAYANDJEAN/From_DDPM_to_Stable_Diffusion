import os
from typing import Dict
import torch
from timeit import default_timer as timer
from torchvision.utils import save_image
from models import VanillaVAE, VQVAE
from utils import animal_faces_loader, denormalize
from torchvision import transforms


def train(config: Dict):
    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')
    dataloader = animal_faces_loader('val', config['batch_size'], config['img_size'])
    train_loader = animal_faces_loader('train', config['batch_size'], config['img_size'])
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
    min_train_loss = float('inf')
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

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(vqvae.state_dict(), os.path.join(config['model_dir'], f'vqvae.pth'))

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, time: {(end_time - start_time):.3f}s")

        _, batch_image = next(enumerate(train_loader))
        images = batch_image[0].to(device)
        latent = vqvae.encode(images)
        reconstruction = vqvae.decode(latent[0])
        result = torch.cat((images, reconstruction), dim=0)
        result = transforms.Resize((128, 128))(result)
        save_image(tensor=denormalize(result),
                   fp=f'../00_assets/image/vae_raw_recons_{epoch}.png',
                   nrow=config['batch_size'],
                   padding=0)


if __name__ == '__main__':
    modelConfig = {
        'epoch': 40,
        'epoch_awoken': None,
        'batch_size': 7,
        'embedding_dim': 4,
        'num_embeddings': 128,
        'lr': 3 * 1e-4,
        'img_channel': 3,
        'img_size': 512,
        'hidden_dims': [32, 64, 128],
        'model_dir': '../00_assets/model_vae/'
    }

    os.makedirs(modelConfig['model_dir'], exist_ok=True)

    train(modelConfig)
