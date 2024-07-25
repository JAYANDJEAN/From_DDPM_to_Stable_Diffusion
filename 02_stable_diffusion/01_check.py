from stable_diffusion_pytorch import tokenizer


def check_tokenizer():
    token = tokenizer.Tokenizer()
    text = "a photograph of an astronaut riding a horse"
    print(token.encode(text))
    print(token.bos_token)
    print(token.eos_token)
    print(token.max_length)


if __name__ == '__main__':
    check_tokenizer()
