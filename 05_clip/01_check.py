import torch
import clip
from PIL import Image
import numpy as np
from model_clip_vit import CLIP


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


def check_clip_model():
    # opened_file = "../00_assets/model_clip/ViT-L-14.pt"
    # model = torch.jit.load(opened_file, map_location="cpu")
    # state_dict = model.state_dict()
    image = torch.randn(5, 3, 224, 224)
    text = torch.randint(low=0, high=100, size=(5, 77))
    model = CLIP(embed_dim=768, image_resolution=224, vision_layers=24,
                 vision_width=1024, vision_patch_size=14, context_length=77,
                 vocab_size=49408, transformer_width=768, transformer_heads=12,
                 transformer_layers=12)

    image_features = model.encode_image(image)
    print(f"image_features shape: {image_features.shape}")
    text_features = model.encode_text(text)
    print(f"text_features shape: {text_features.shape}")


if __name__ == '__main__':
    check_clip_model()
