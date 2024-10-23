import replicate
import base64
import cv2

# model_id = "stability-ai/stable-diffusion-3.5-large"
# model_id = "batouresearch/sdxl-controlnet-lora:3bb13fe1c33c35987b33792b01b71ed6529d03f165d1c2416375859f09ca9fef"
model_id = "lucataco/sdxl-controlnet:06d6fae3b75ab68a28cd2900afa6033166910dd09fd9751047043a5bbb4c184b"


def get_input(file_path):
    with open(file_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
    init_image = f"data:application/octet-stream;base64,{base64_encoded}"
    input = {
        "image": init_image,
        "prompt": "a cat wearing a denim jacket.",
        "condition_scale": 0.6,
        "negative_prompt": "low quality, bad quality, sketches",
        "num_inference_steps": 50
    }
    return input


output = replicate.run(
    model_id,
    input=get_input('../00_assets/image/cat.png')
)
print(output)
