"""Dependency-free CLIP ViT-B/16 vision tower (torch only), shared by training and the
Codabench submission so both run the exact same forward pass."""

import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class EncoderLayer(nn.Module):
    def __init__(self, width=768, heads=12, eps=1e-5, act='quick_gelu'):
        super().__init__()
        self.heads, self.act = heads, act
        self.layer_norm1 = nn.LayerNorm(width, eps=eps)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.layer_norm2 = nn.LayerNorm(width, eps=eps)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        b, t, w = x.shape
        h = self.layer_norm1(x)
        q, k, v = (proj(h).view(b, t, self.heads, w // self.heads).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        x = x + self.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, w))
        h = self.fc1(self.layer_norm2(x))
        h = (h * torch.sigmoid(1.702 * h) if self.act == 'quick_gelu' else F.gelu(h) if self.act == 'gelu'
             else F.gelu(h, approximate='tanh'))
        return x + self.fc2(h)


class CLIPVision(nn.Module):
    """Returns (pooled CLS after post_layernorm, projected 512-d embedding)."""

    grad_ckpt = False  # train.py --grad-ckpt: recompute layer activations in backward (same maths, less memory)

    def __init__(self, size=(224, 224), width=768, layers=12, patch=16, embed=512, act='quick_gelu', heads=None):
        super().__init__()
        self.grid = (size[0] // patch, size[1] // patch)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.patch_embedding = nn.Conv2d(3, width, patch, patch, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1 + self.grid[0] * self.grid[1], width))
        self.pre_layrnorm = nn.LayerNorm(width, eps=1e-5)
        self.layers = nn.ModuleList(EncoderLayer(width, heads or width // 64, act=act)  # 12 heads (B), 16 (L, H)
                                    for _ in range(layers))
        self.post_layernorm = nn.LayerNorm(width, eps=1e-5)
        self.visual_projection = nn.Linear(width, embed, bias=False)

    def load_hf_state(self, state):
        """Load a Hugging Face CLIPModel / CLIPVisionModel state dict (or our own)."""
        mapped = {}
        for name, tensor in state.items():
            name = name.removeprefix('vision_model.')
            name = (name.replace('embeddings.class_embedding', 'class_embedding')
                        .replace('embeddings.patch_embedding', 'patch_embedding')
                        .replace('embeddings.position_embedding.weight', 'position_embedding')
                        .replace('encoder.layers.', 'layers.')
                        .replace('.self_attn.', '.').replace('.mlp.', '.'))
            mapped[name] = tensor
        pos = mapped['position_embedding']
        if pos.shape != self.position_embedding.shape:
            mapped['position_embedding'] = resize_position_embedding(pos, self.grid)
        own = self.state_dict()
        proj = 'visual_projection.weight'  # only the 'text' head uses it: keep the init if absent or another size
        if proj not in mapped or mapped[proj].shape != own[proj].shape:
            mapped[proj] = own[proj]
        missing = sorted(set(own) - set(mapped))
        if missing:
            raise RuntimeError(f'Missing CLIP parameters: {missing}')
        self.load_state_dict({k: mapped[k] for k in own}, strict=True)

    def tokens(self, pixels, stop=None):
        """Patch + position embedding and encoder layers [0, stop)."""
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + self.position_embedding.to(x.dtype)
        x = self.pre_layrnorm(x)
        for layer in self.layers[:stop]:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def pool(self, x):
        """Final tokens -> (post-LN CLS, projected embedding)."""
        pooled = self.post_layernorm(x[:, 0])
        return pooled, self.visual_projection(pooled)

    def tail(self, x, start):
        """Encoder layers [start, end) + pooling."""
        for layer in self.layers[start:]:
            x = layer(x)
        return self.pool(x)

    def forward(self, pixels):
        return self.tail(self.tokens(pixels), len(self.layers))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))


class MAPHead(nn.Module):
    """SigLIP attention-pooling head: one learned probe attends over all tokens."""

    def __init__(self, width=768, heads=12):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.layernorm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = MLP(width)

    def forward(self, x):
        h = self.attention(self.probe.expand(x.shape[0], -1, -1).to(x.dtype), x, x, need_weights=False)[0]
        return (h + self.mlp(self.layernorm(h)))[:, 0]


class SiglipVision(nn.Module):
    """SigLIP / SigLIP 2 (fixed-resolution) ViT-B/16 vision tower with the CLIPVision interface. Inputs stay
    CLIP-normalised (same preprocessing everywhere); they are re-normalised to SigLIP's mean 0.5 / std 0.5 inside.
    No CLS token: pooled = attention-pooling head over the post-LN tokens; the 'embedding' is the pooled vector."""

    def __init__(self, size=(256, 128), width=768, layers=12, patch=16):
        super().__init__()
        self.grid = (size[0] // patch, size[1] // patch)
        self.patch_embedding = nn.Conv2d(3, width, patch, patch)
        self.position_embedding = nn.Parameter(torch.zeros(self.grid[0] * self.grid[1], width))
        self.layers = nn.ModuleList(EncoderLayer(width, width // 64, 1e-6, 'gelu_tanh') for _ in range(layers))
        self.post_layernorm = nn.LayerNorm(width, eps=1e-6)
        self.head = MAPHead(width, width // 64)
        self.visual_projection = nn.Identity()
        std, mean = torch.tensor(CLIP_STD).view(1, 3, 1, 1), torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.register_buffer('in_scale', std / 0.5, persistent=False)
        self.register_buffer('in_shift', (mean - 0.5) / 0.5, persistent=False)

    def load_hf_state(self, state):
        """Load a Hugging Face SiglipModel / SiglipVisionModel state dict (or our own)."""
        mapped = {}
        for name, tensor in state.items():
            if name.startswith('text_model.') or name in ('logit_scale', 'logit_bias'):
                continue
            name = name.removeprefix('vision_model.')
            name = (name.replace('embeddings.patch_embedding', 'patch_embedding')
                        .replace('embeddings.position_embedding.weight', 'position_embedding')
                        .replace('encoder.layers.', 'layers.'))
            if name.startswith('layers.'):
                name = name.replace('.self_attn.', '.').replace('.mlp.', '.')
            mapped[name] = tensor
        naflex = mapped['patch_embedding.weight'].ndim == 2
        if naflex:  # SigLIP 2 NaFlex: Linear over (row, col, channel)-flattened patches -> equivalent Conv2d
            w = mapped['patch_embedding.weight']
            k = self.patch_embedding.kernel_size[0]
            mapped['patch_embedding.weight'] = w.view(w.shape[0], k, k, 3).permute(0, 3, 1, 2).contiguous()
        pos = mapped['position_embedding']
        if pos.shape != self.position_embedding.shape:  # NaFlex resizes its grid bilinearly with antialias
            mapped['position_embedding'] = resize_position_embedding(pos, self.grid, cls=False,
                                                                     mode='bilinear' if naflex else 'bicubic')
        own = self.state_dict()
        missing = sorted(set(own) - set(mapped))
        if missing:
            raise RuntimeError(f'Missing SigLIP parameters: {missing}')
        self.load_state_dict({k: mapped[k] for k in own}, strict=True)

    def tokens(self, pixels, stop=None):
        pixels = pixels * self.in_scale.to(pixels.dtype) + self.in_shift.to(pixels.dtype)
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2) + self.position_embedding.to(pixels.dtype)
        for layer in self.layers[:stop]:
            x = layer(x)
        return x

    def pool(self, x):
        pooled = self.head(self.post_layernorm(x))
        return pooled, pooled

    def tail(self, x, start):
        for layer in self.layers[start:]:
            x = layer(x)
        return self.pool(x)

    def forward(self, pixels):
        return self.tail(self.tokens(pixels), len(self.layers))


def resize_position_embedding(pos, grid, cls=True, mode='bicubic'):
    """Bicubic resize of the patch position grid (CLS kept if present), e.g. 14x14 -> 16x8."""
    cls, patches = (pos[:1], pos[1:]) if cls else (pos[:0], pos)
    side = int(math.isqrt(len(patches)))
    patches = patches.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=grid, mode=mode, align_corners=False, antialias=mode == 'bilinear')
    patches = patches.permute(0, 2, 3, 1).reshape(grid[0] * grid[1], -1).to(pos.dtype)
    return torch.cat((cls, patches))


class AttributeHead(nn.Module):
    """'linear': LayerNorm -> Linear on pooled CLS.
    'text': cosine similarity of the projected image embedding to 40 learnable
    attribute embeddings initialised from CLIP text prompts, per-attribute bias."""

    def __init__(self, kind='linear', width=768, embed=512, classes=40):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, classes))
        else:
            self.text = nn.Parameter(torch.zeros(classes, embed))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
            self.bias = nn.Parameter(torch.zeros(classes))

    def forward(self, pooled, projected):
        if self.kind == 'linear':
            return self.net(pooled)
        image = F.normalize(projected, dim=-1)
        text = F.normalize(self.text, dim=-1)
        return self.logit_scale.exp() * image @ text.t() + self.bias


class PARModel(nn.Module):
    def __init__(self, size=(256, 128), head='linear', backbone='clip'):
        super().__init__()
        self.vision = SiglipVision(size) if backbone == 'siglip' else CLIPVision(size)
        self.head = AttributeHead(head, embed=768) if backbone == 'siglip' else AttributeHead(head)

    def forward(self, pixels):
        return self.head(*self.vision(pixels))
