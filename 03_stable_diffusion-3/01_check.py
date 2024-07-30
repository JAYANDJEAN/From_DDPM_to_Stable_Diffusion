import torch
from diffusers import StableDiffusion3Pipeline


class SDTokenizer:
    def __init__(self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True,
                 min_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str):
        """
        Tokenize the text, with weight values - presume 1.0 for all and ignore other features here.
        The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3.
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


def check_pipeline():
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id,
                                                    torch_dtype=torch.float16,
                                                    token="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")
    # pipe = pipe.to("cuda")

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    image.save("output.png")


def check_clip():
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    inputs = tokenizer(["a"], padding=True, return_tensors="pt")
    print(inputs)
    outputs = model(**inputs)

    text_embeds = outputs.text_embeds
    print(text_embeds[0, 0:5])
    model = CLIPTextModelWithProjection.from_pretrained("models/clip_l")
    with torch.no_grad():
        if hasattr(model, 'text_projection') and hasattr(model.text_projection, 'weight'):
            model.text_projection.weight.copy_(torch.eye(768))
        else:
            print("text_projection.weight not found in model state_dict")
    outputs = model(**inputs)
    text_embeds = outputs.text_embeds
    print(text_embeds[0, 0:5])

    old_tensor = torch.load('tensor_out_cat.pt')
    print(old_tensor.shape)
    print(old_tensor[0, 0:10, 0:5])


if __name__ == '__main__':
    check_clip()
