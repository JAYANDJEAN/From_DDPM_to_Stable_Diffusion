import torch
from vit_pytorch import ViT

import os
from typing import Dict
from timeit import default_timer as timer
from torch.nn import CrossEntropyLoss
from utils import animal_faces_loader


def train(config: Dict):
    print('Model Training...............')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use device: {device}')
    test_loader = animal_faces_loader('val', config['batch_size'], config['image_size'])
    train_loader = animal_faces_loader('train', config['batch_size'], config['image_size'])
    vit = ViT(image_size=config['image_size'], patch_size=config['patch_size'],
              num_classes=config['num_classes'], dim=config['dim'],
              depth=config['depth'], heads=config['heads'], mlp_dim=config['mlp_dim'],
              dropout=config['dropout'], emb_dropout=config['emb_dropout'],
              ).to(device)
    print('Total trainable parameters:', sum(p.numel() for p in vit.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(vit.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = CrossEntropyLoss()
    min_train_loss = float('inf')

    for epoch in range(config['epoch']):
        start_time = timer()
        losses = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            result = vit(images)
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(list(train_loader))
        end_time = timer()

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(vit.state_dict(), os.path.join(config['model_dir'], f'vqvae.pth'))

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, time: {(end_time - start_time):.3f}s")

        vit.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                predicted = torch.argmax(vit(images), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch: {epoch}, Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    model_config = {
        'epoch': 10,
        'batch_size': 32,
        'image_size': 128,
        'patch_size': 32,
        'num_classes': 3,
        'dim': 512,
        'depth': 6,
        'heads': 8,
        'mlp_dim': 1024,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'model_dir': '../00_assets/model_vit/'
    }

    os.makedirs(model_config['model_dir'], exist_ok=True)

    train(model_config)
