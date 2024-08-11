import torch
import clip
from PIL import Image
import numpy as np


def check_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    image_path = "../00_assets/datasets/afhq/train/cat/flickr_cat_000014.jpg"
    image = Image.open(image_path)
    print(f"image shape before preprocess: {np.array(image).shape}")
    image = preprocess(image).unsqueeze(0).to(device)
    print(f"image shape after preprocess: {image.shape}")
    text = clip.tokenize(["a dog", "a cat"]).to(device)
    print(f"text shape: {text.shape}")

    with torch.no_grad():
        image_features = model.encode_image(image)
        print(f"image_features shape: {image_features.shape}")
        text_features = model.encode_text(text)
        print(f"text_features shape: {text_features.shape}")
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)


if __name__ == '__main__':
    check_clip()
