from safetensors import safe_open

# with safe_open("models/clip_l/model.safetensors", framework="pt", device='cpu') as f:
#     for k in f.keys():
#         print(k)
#         tensor = f.get_tensor(k)
#         print(tensor.shape)


from transformers import AutoTokenizer, CLIPTextModelWithProjection

model = CLIPTextModelWithProjection.from_pretrained("./models/clip_l")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
text_embeds = outputs.text_embeds
print(text_embeds.shape)