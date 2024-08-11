import torch, math, einops
from torch import nn
from transformers import CLIPTokenizer, T5TokenizerFast

ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    print("sample_euler")
    print(f"s_in: {s_in}, shape: {s_in.shape}")
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        debug = True if i == 0 else False
        denoised = model(x, sigma_hat * s_in, debug=debug, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype,
                              device=device)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None,
                 device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        # self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        self.mlp = Mlp(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation],
                       dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in
             range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = torch.nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype,
                                   device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[
            torch.arange(x.shape[0], device=x.device), input_tokens.to(dtype=torch.int, device=x.device).argmax(
                dim=-1),]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


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
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here. The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
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


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text: str):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        return out


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self, device="cpu", max_length=77, layer="last", layer_idx=None, textmodel_json_config=None,
                 dtype=None, model_class=CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True,
                 return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = torch.LongTensor(tokens).to(device)
        outputs = self.transformer(tokens, intermediate_output=self.layer_idx,
                                   final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype,
                         special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""

    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype,
                         special_tokens={"end": 1, "pad": 0}, model_class=T5)


class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""

    def __init__(self):
        super().__init__(pad_with_end=False, tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
                         has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=77)


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=x.device, dtype=x.dtype) * x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads,
                                                              device=device)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large,
                                               torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        out = attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, vocab_size, dtype, device):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList([T5Block(model_dim, inner_dim, ff_dim, num_heads,
                                                  relative_attention_bias=(i == 0), dtype=dtype, device=device) for i in
                                          range(num_layers)])
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.embed_tokens(input_ids)
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        self.encoder = T5Stack(self.num_layers, config_dict["d_model"], config_dict["d_model"], config_dict["d_ff"],
                               config_dict["num_heads"], config_dict["vocab_size"], dtype, device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                     device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                     device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                                dtype=dtype, device=device)
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype,
                                 device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype,
                                 device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype,
                                 device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype,
                                        device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, dtype=dtype,
                                    device=device)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                    device=device)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=16,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                        device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, resolution=256, z_channels=16,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                       device=device)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype,
                                        device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast("cuda", dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)
