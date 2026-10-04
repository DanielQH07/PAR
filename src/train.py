"""PAR training: one leave-one-domain-out fold of CLIP ViT-B/16 fine-tuning,
scored at the end by prepare.evaluate_heldout (fixed). Seeded from src/train_dg_2027.py
(recipe of submission v6: 4 unfrozen blocks, linear head).

Each switch targets one weakness of the partial fine-tune baseline:
  --size H W            pedestrian-shaped input instead of padding to a square (W5)
  --aug strong          geometry / resolution / blur / erasing, no hue jitter (W6)
  --domain-balance a    sampler weight n_domain^-a                               (W7)
  --mask-missing        drop loss on (domain, attribute) pairs never annotated   (W7)
  --pos-power p         pos_weight = (neg/pos)^p, p=1 is the old loss            (W2)
  --label-smoothing e                                                            (W2)
  --head text           cosine head initialised from CLIP text prompts           (W8)
  --lp-epochs n         head-only epochs before unfreezing (LP-FT)               (W3)
  --ema d, cosine lr, weight decay; WiSE-FT alphas evaluated at the end          (W3)
  --domain-tags p       NMT-style domain tags: per-domain logit bias, replaced by a
                        'generic' tag with prob p; test uses the generic tag only
  --anchor b            reference anchoring (like the KL-to-reference term in RLHF):
                        b * (1 - cos) between fine-tuned and frozen pre-trained CLIP embeddings
  --crowd-colors r      crowd layer (Rodrigues & Pereira, AAAI 2018) for colour groups: per-source
                        12x12 transforms of the upper/lower colour logits, identity-initialised,
                        L2 weight r towards identity, dropped at test time
Checkpoints are selected with the Codabench metric on source-domain validation (W1).
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from clip_vit import CLIP_MEAN, CLIP_STD, PARModel
from evaluate_cached_head import challenge_score
import prepare

PROMPTS = {  # attribute -> phrase completing "a photo of a person ..."
    'Age-Young': 'who is a child or teenager', 'Age-Adult': 'who is an adult',
    'Age-Old': 'who is elderly', 'Gender-Female': 'who is a woman',
    'Hair-Length-Short': 'with short hair', 'Hair-Length-Long': 'with long hair',
    'Hair-Length-Bald': 'who is bald', 'UpperBody-Length-Short': 'wearing a short-sleeved top',
    'LowerBody-Length-Short': 'wearing shorts or a short skirt',
    'LowerBody-Type-Trousers&Shorts': 'wearing trousers or shorts',
    'LowerBody-Type-Skirt&Dress': 'wearing a skirt or a dress',
    'Accessory-Backpack': 'carrying a backpack', 'Accessory-Bag': 'carrying a bag',
    'Accessory-Glasses-Normal': 'wearing glasses', 'Accessory-Glasses-Sun': 'wearing sunglasses',
    'Accessory-Hat': 'wearing a hat',
}


def prompt(name):
    if name in PROMPTS:
        return f'a photo of a person {PROMPTS[name]}.'
    part, _, color = name.split('-')  # UpperBody-Color-Red
    color = 'unusually colored' if color == 'Other' else color.lower()
    garment = 'top' if part == 'UpperBody' else 'trousers or skirt'
    return f'a photo of a person wearing a {color} {garment}.'


def read_rows(path):
    with open(path, newline='') as stream:
        rows = list(csv.reader(stream))
    return rows[0][1:], rows[1:]


class Images(Dataset):
    def __init__(self, rows, root, size, train=False, aug='flip'):
        self.paths = np.array([r[0] for r in rows])  # compact: avoids per-worker list copies
        self.labels = np.asarray([r[1:] for r in rows], dtype=np.float32)
        self.root, self.size, self.train, self.aug = Path(root), size, train, aug
        self.mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(3, 1, 1)
        self.pad = tuple(round(c * 255) for c in CLIP_MEAN)
        self.groups = None
        self.wb, self.gamma = 0.0, 0.0  # W1: camera colour-cast / exposure jitter (training only)

    def set_identity_pairs(self):
        """P1-id: Market file names carry the person id; each item also returns another image of the same person."""
        ids = {}
        for i, p in enumerate(self.paths):
            if p.startswith('Market1501/'):
                ids.setdefault(p.rsplit('/', 1)[-1].split('_')[0], []).append(i)
        self.groups = {i: g for g in ids.values() if len(g) > 1 for i in g}
        return len(self.groups)

    def set_identity_index(self):
        """P3: per-image Market person index (-1 elsewhere); items return it as a 4th field."""
        ids, self.person = {}, np.full(len(self.paths), -1, dtype=np.int64)
        for i, p in enumerate(self.paths):
            if p.startswith('Market1501/'):
                self.person[i] = ids.setdefault(p.rsplit('/', 1)[-1].split('_')[0], len(ids))
        return len(ids)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x, y, d = self.load(index), self.labels[index], domain_of(str(self.paths[index]))
        if getattr(self, 'with_index', False):  # L2: ELR keeps a per-image target
            return x, y, d, index
        if getattr(self, 'person', None) is not None:
            return x, y, d, self.person[index]
        if self.groups is None:
            return x, y, d
        g = self.groups.get(index)
        if g is None:
            return x, y, d, x, y, False
        j = random.choice([k for k in g if k != index])
        return x, y, d, self.load(j), self.labels[j], True

    def load(self, index):
        path = str(self.paths[index])
        h, w = self.size
        with Image.open(self.root / path) as image:
            image = image.convert('RGB')
            strong = self.train and self.aug == 'strong'
            if strong and random.random() < 0.5:  # crop 85-100% of each side
                sw, sh = (random.uniform(0.85, 1.0) for _ in range(2))
                cw, ch = max(1, round(image.width * sw)), max(1, round(image.height * sh))
                x0, y0 = random.randint(0, image.width - cw), random.randint(0, image.height - ch)
                image = image.crop((x0, y0, x0 + cw, y0 + ch))
            if strong and random.random() < 0.3:  # simulate low-resolution cameras
                f = random.uniform(0.25, 0.6)
                small = (max(8, round(image.width * f)), max(8, round(image.height * f)))
                image = image.resize(small, Image.Resampling.BILINEAR)
            if strong and random.random() < 0.2:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
            scale = min(w / image.width, h / image.height)
            new = (min(w, max(1, round(image.width * scale))), min(h, max(1, round(image.height * scale))))
            image = image.resize(new, Image.Resampling.BICUBIC)
            if self.train and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            canvas = Image.new('RGB', (w, h), self.pad)
            canvas.paste(image, ((w - image.width) // 2, (h - image.height) // 2))
            x = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255
        if strong:  # brightness/contrast only: colours are labels
            x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5) * random.uniform(0.8, 1.2)
            x = x.clamp(0, 1)
        if self.train and self.wb:  # per-channel gains: white balance of another camera (hue order kept)
            x = (x * (1 + self.wb * (2 * torch.rand(3, 1, 1) - 1))).clamp(0, 1)
        if self.train and self.gamma:  # exposure / tone curve
            x = x ** math.exp(self.gamma * (2 * random.random() - 1))
        x = (x - self.mean) / self.std
        if strong and random.random() < 0.3:  # random erasing
            eh, ew = random.randint(h // 8, h // 3), random.randint(w // 8, w // 3)
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            x[:, y0:y0 + eh, x0:x0 + ew] = torch.randn(3, eh, ew)
        return x


class PKBatches:
    """Q1 stage 2: each batch = P Market people x K images each + P*K images drawn with the domain-balanced weights.
    One epoch = one pass over the people (len = people // P)."""

    def __init__(self, person, weight, p, k):
        groups = {}
        for i, q in enumerate(person):
            if q >= 0:
                groups.setdefault(int(q), []).append(i)
        self.groups = [g for g in groups.values() if len(g) >= 2]
        self.weight, self.p, self.k = weight / weight.sum(), p, k

    def __len__(self):
        return len(self.groups) // self.p

    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            for gi in np.random.choice(len(self.groups), self.p, replace=False):
                g = self.groups[gi]
                batch += np.random.choice(g, self.k, replace=len(g) < self.k).tolist()
            batch += np.random.choice(len(self.weight), self.p * self.k, p=self.weight).tolist()
            yield batch


DOMAINS = ('Market1501', 'PA100k', 'PETA')
GENERIC = len(DOMAINS)  # index of the domain-agnostic tag


class QueryHead(torch.nn.Module):
    """H1: attribute queries cross-attend to the patch tokens (ML-Decoder style: no self-attention,
    one query per attribute, group FC), with a query-conditioned sigmoid gate on the attention output
    (gated attention from LLMs). Output weights start at zero, so the model starts as the CLS head."""

    def __init__(self, classes=40, width=768, heads=8, gate=True, queries=None):
        super().__init__()
        self.queries = torch.nn.Parameter(torch.randn(queries or classes, width) * 0.02)
        self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.gate = torch.nn.Linear(width, width) if gate else None
        self.norm = torch.nn.LayerNorm(width)
        self.out_w = torch.nn.Parameter(torch.zeros(classes, width))
        self.out_b = torch.nn.Parameter(torch.zeros(classes))

    def forward(self, patches):
        q = self.queries.to(patches.dtype).expand(len(patches), -1, -1)
        o, _ = self.attn(q, patches, patches, need_weights=False)
        if self.gate is not None:
            o = o * torch.sigmoid(self.gate(q))
        return (self.norm(o) * self.out_w).sum(-1) + self.out_b  # 1 query broadcasts to all attributes


BANDS = {'head': (0, 4), 'upper': (3, 10), 'lower': (8, 16)}  # rows of the 16x8 patch grid at 256x128


def band_of(name):
    if name.startswith(('Hair', 'Accessory-Hat', 'Accessory-Glasses')):
        return 'head'
    if name.startswith(('UpperBody', 'Accessory-Backpack', 'Accessory-Bag')):
        return 'upper'
    if name.startswith('LowerBody'):
        return 'lower'
    return None  # age, gender: whole body


class RegionHead(torch.nn.Module):
    """H2: learnable per-attribute spatial prior over patch positions (softmax pooling), initialised to
    body bands (SSPNet-like spatial prior, content-independent, ~0 cost). Zero-init output."""

    def __init__(self, names, grid, width=768):
        super().__init__()
        rows, cols = grid
        prior = torch.zeros(len(names), rows, cols)
        for i, n in enumerate(names):
            band = band_of(n)
            if band:
                lo, hi = BANDS[band]
                prior[i] = -4.0
                prior[i, lo:min(hi, rows)] = 0.0
        self.prior = torch.nn.Parameter(prior.flatten(1))
        self.norm = torch.nn.LayerNorm(width)
        self.out_w = torch.nn.Parameter(torch.zeros(len(names), width))
        self.out_b = torch.nn.Parameter(torch.zeros(len(names)))

    def forward(self, patches):
        w = torch.softmax(self.prior, -1).to(patches.dtype)
        feats = torch.einsum('an,bnd->bad', w, patches)
        return (self.norm(feats) * self.out_w).sum(-1) + self.out_b


class LoRALinear(torch.nn.Module):
    """Q2: frozen Linear + trainable low-rank update (W + scale * B A); merge() folds it back for export/inference."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base.requires_grad_(False), alpha / rank
        self.lora_a = torch.nn.Parameter(torch.randn(rank, base.in_features) / base.in_features ** 0.5)
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scale

    def merged(self):
        lin = torch.nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + self.scale * self.lora_b.data @ self.lora_a.data
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


def add_lora(layers, rank, alpha):
    for layer in layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'):
            setattr(layer, name, LoRALinear(getattr(layer, name), rank, alpha))


def merge_lora(module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged())
        else:
            merge_lora(child)


class TimmPEVision(torch.nn.Module):
    """P1-pe: Perception Encoder PE-Core B/16 (timm EVA with 2D RoPE, MAP pooling) with the CLIPVision interface;
    training only (a submission would ship it as TorchScript). Inputs stay CLIP-normalised, re-normalised to 0.5/0.5."""

    def __init__(self, size):
        super().__init__()
        import timm
        self.m = timm.create_model('vit_pe_core_base_patch16_224', pretrained=False, img_size=tuple(size), num_classes=1024)
        self.layers, self.post_layernorm = self.m.blocks, self.m.norm
        self.visual_projection = torch.nn.Identity()
        self.head = torch.nn.ModuleList([self.m.attn_pool, self.m.fc_norm])
        std, mean = torch.tensor(CLIP_STD).view(1, 3, 1, 1), torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self.register_buffer('in_scale', std / 0.5, persistent=False)
        self.register_buffer('in_shift', (mean - 0.5) / 0.5, persistent=False)
        self._rope = None

    def load_hf_state(self, state):
        from timm.models.eva import checkpoint_filter_fn
        missing = self.m.load_state_dict(checkpoint_filter_fn(state, self.m), strict=False).missing_keys
        assert not missing, missing

    def tokens(self, pixels, stop=None):
        x = self.m.patch_embed(pixels * self.in_scale.to(pixels.dtype) + self.in_shift.to(pixels.dtype))
        x, self._rope = self.m._pos_embed(x)
        x = self.m.norm_pre(x)
        for blk in self.layers[:stop]:
            x = blk(x, rope=self._rope)
        return x

    def pool(self, x):
        pooled = self.m.forward_head(self.post_layernorm(x), pre_logits=True)
        return pooled, pooled

    def tail(self, x, start):
        for blk in self.layers[start:]:
            x = blk(x, rope=self._rope)
        return self.pool(x)


class PARPlus(PARModel):
    """PARModel + optional patch-token head added to the CLS logits + optional logit soft-capping
    (c * tanh(z / c), Gemma-2 style)."""

    def __init__(self, size, head, extra=None, names=None, softcap=0.0, arch='b16'):
        super().__init__(size, head)
        if arch == 'l14':  # B1 teacher: CLIP ViT-L/14 (24 layers, width 1024, 16 heads, 768-d projection)
            from clip_vit import AttributeHead, CLIPVision
            self.vision = CLIPVision(size, width=1024, layers=24, patch=14, embed=768)
            self.head = AttributeHead(head, width=1024, embed=768)
        elif arch == 'pe':
            from clip_vit import AttributeHead
            self.vision = TimmPEVision(size)
            self.head = AttributeHead(head, embed=768)
        elif arch == 'siglip':  # S1: SigLIP 2 ViT-B/16 (attention-pooling head, no CLS)
            from clip_vit import AttributeHead, SiglipVision
            self.vision = SiglipVision(size)
            self.head = AttributeHead(head, embed=768)
        grid = (size[0] // 16, size[1] // 16)
        self.extra = (QueryHead() if extra == 'query' else QueryHead(gate=False) if extra == 'query-nogate'
                      else QueryHead(queries=1) if extra == 'query1'
                      else RegionHead(names, grid) if extra == 'region' else None)
        self.softcap = softcap
        self.mixstyle_layers, self.mixstyle_p = (), 0.5
        self.fuse_layers, self.fuse = (), None
        self.attr_from = None

    def set_attr_tokens(self, k, text=None):
        """V1 (VTB / PromptPAR-like image-text fusion): 40 attribute tokens join the last k blocks. Image tokens do not
        attend to them (the image path, hence the CLS head, is unchanged); each attribute token reads the image tokens
        and gives one logit through a zero-initialised readout. Tokens start from the attribute's CLIP text embedding
        mapped back through the visual projection (text: (40, embed)), else from the class embedding."""
        n, width = len(self.vision.layers), self.vision.class_embedding.shape[0]
        self.attr_from = n - k
        init = self.vision.class_embedding.detach().expand(40, -1).clone()
        if text is not None:
            t = torch.nn.functional.normalize(text.float(), dim=-1) @ self.vision.visual_projection.weight.detach()
            init = init + t / t.norm(dim=-1, keepdim=True) * self.vision.class_embedding.detach().norm()
        self.attr_tokens = torch.nn.Parameter(init)
        self.attr_w = torch.nn.Parameter(torch.zeros(40, width))
        self.attr_b = torch.nn.Parameter(torch.zeros(40))

    def set_fuse(self, layers, classes=40):
        """F1: add CLS tokens of intermediate blocks (1-based) through a zero-initialised linear head."""
        self.fuse_layers = tuple(layers)
        width = self.vision.class_embedding.shape[0]
        self.fuse = torch.nn.Sequential(torch.nn.LayerNorm(width * len(layers)), torch.nn.Linear(width * len(layers), classes))
        torch.nn.init.zeros_(self.fuse[1].weight); torch.nn.init.zeros_(self.fuse[1].bias)

    def mixstyle(self, x):
        """MixStyle (Zhou et al. 2021) on ViT tokens: mix per-sample token mean/std across the batch (training only)."""
        if not self.training or torch.rand(()) > self.mixstyle_p:
            return x
        mu, sig = x.mean(1, keepdim=True), (x.var(1, keepdim=True) + 1e-6).sqrt()
        mu, sig = mu.detach(), sig.detach()
        lam = torch.distributions.Beta(0.1, 0.1).sample((len(x), 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(len(x), device=x.device)
        return (x - mu) / sig * (lam * sig + (1 - lam) * sig[perm]) + lam * mu + (1 - lam) * mu[perm]

    def forward(self, pixels):
        mids = []
        a = None
        if (self.mixstyle_layers and self.training) or self.fuse_layers or self.attr_from is not None:
            x = self.vision.tokens(pixels, 0)
            for i, layer in enumerate(self.vision.layers):
                if i == self.attr_from:
                    t = x.shape[1]
                    x = torch.cat((x, self.attr_tokens.to(x.dtype).expand(len(x), -1, -1)), 1)
                    mask = torch.ones(t + 40, t + 40, dtype=torch.bool, device=x.device)
                    mask[:t, t:] = False  # image tokens never see the attribute tokens
                x = masked_layer(layer, x, mask) if self.attr_from is not None and i >= self.attr_from else layer(x)
                if self.training and i + 1 in self.mixstyle_layers:
                    x = self.mixstyle(x)
                if i + 1 in self.fuse_layers:
                    mids.append(x[:, 0])
            if self.attr_from is not None:
                x, a = x[:, :-40], x[:, -40:]
        else:
            x = self.vision.tokens(pixels)
        self.last_tokens = x  # R1: final hidden tokens for feature distillation
        pooled, self.last_embed = self.vision.pool(x)  # last_embed: used by the caption loss (T1)
        z = self.head(pooled, self.last_embed)
        if self.extra is not None:
            z = z + self.extra(self.vision.post_layernorm(x[:, 1:]))
        if self.fuse is not None:
            z = z + self.fuse(torch.cat(mids, -1))
        if a is not None:
            z = z + (self.vision.post_layernorm(a) * self.attr_w).sum(-1) + self.attr_b
        return self.softcap * torch.tanh(z / self.softcap) if self.softcap else z


def lipsum_bank(weights, n=4096, length=8):
    """Lipsum-FT (Nam et al., ICLR 2024): normalised CLIP text features of n random token sequences."""
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(weights, local_files_only=True).eval()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 49406, (n, length), generator=g)  # 49406/49407 = BOS/EOS of the CLIP tokenizer
    ids = torch.cat((torch.full((n, 1), 49406), ids, torch.full((n, 1), 49407)), 1)
    with torch.no_grad():
        feats = torch.cat([model.get_text_features(input_ids=ids[i:i + 512]) for i in range(0, n, 512)])
    return torch.nn.functional.normalize(feats.float(), dim=-1)


def masked_layer(layer, x, mask):
    """clip_vit.EncoderLayer.forward with a boolean attention mask (True = may attend)."""
    b, t, w = x.shape
    h = layer.layer_norm1(x)
    q, k, v = (proj(h).view(b, t, layer.heads, w // layer.heads).transpose(1, 2)
               for proj in (layer.q_proj, layer.k_proj, layer.v_proj))
    x = x + layer.out_proj(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                           .transpose(1, 2).reshape(b, t, w))
    h = layer.fc1(layer.layer_norm2(x))
    return x + layer.fc2(h * torch.sigmoid(1.702 * h))


def newton_schulz(g, steps=5):
    """Approximate orthogonalisation U V^T of g (Muon, Jordan et al. 2024; quintic coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return (x.T if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D weights with the Moonlight scaling (update * 0.2 * sqrt(max(m, n))), so the update RMS
    matches AdamW and the AdamW learning rate / weight decay can be reused (Liu et al. 2025, Moonlight)."""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault('buf', torch.zeros_like(p.grad))
                buf.mul_(group['momentum']).add_(p.grad)
                g = p.grad.add(buf, alpha=group['momentum'])  # Nesterov
                u = newton_schulz(g) * 0.2 * max(p.shape) ** 0.5
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(u, alpha=-group['lr'])


def domain_of(path):
    return DOMAINS.index(path.split('/')[0])


def text_embeddings(weights, tokenizer, names):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(weights, local_files_only=True).eval()
    tokens = CLIPTokenizer.from_pretrained(tokenizer, local_files_only=True)(
        [prompt(n) for n in names], padding=True, return_tensors='pt')
    with torch.no_grad():
        return model.get_text_features(**tokens)


def caption(y, annotated, names):
    """T1: one sentence describing the person from its (annotated) labels."""
    on = lambda n: n in names and annotated[names.index(n)] and y[names.index(n)] > 0.5
    known = lambda n: n in names and annotated[names.index(n)]
    who = 'woman' if on('Gender-Female') else 'man' if known('Gender-Female') else 'person'
    age = 'young ' if on('Age-Young') else 'elderly ' if on('Age-Old') else 'adult ' if on('Age-Adult') else ''
    hair = ('bald' if on('Hair-Length-Bald') else 'long hair' if on('Hair-Length-Long')
            else 'short hair' if on('Hair-Length-Short') else '')
    colours = lambda part: ' and '.join(n.split('-')[-1].lower().replace('other', 'multicoloured')
                                        for n in names if n.startswith(part + '-Color') and on(n))
    up = colours('UpperBody')
    top = ' '.join(w for w in (up, 'short-sleeved' if on('UpperBody-Length-Short') else 'long-sleeved'
                               if known('UpperBody-Length-Short') else '', 'top') if w)
    low = colours('LowerBody')
    garment = ('skirt or dress' if on('LowerBody-Type-Skirt&Dress')
               else 'shorts' if on('LowerBody-Length-Short') else 'trousers')
    bottom = ' '.join(w for w in (low, garment) if w)
    extras = [t for n, t in (('Accessory-Backpack', 'carrying a backpack'), ('Accessory-Bag', 'carrying a bag'),
                             ('Accessory-Hat', 'wearing a hat'), ('Accessory-Glasses-Sun', 'wearing sunglasses'),
                             ('Accessory-Glasses-Normal', 'wearing glasses')) if on(n)]
    art = 'an' if (age or who)[0] in 'aeiou' else 'a'
    s = f'a photo of {art} {age}{who}' + (f' with {hair}' if hair and hair != 'bald' else ' who is bald' if hair else '')
    return s + f', wearing a {top} and {bottom}' + (', ' + ', '.join(extras) if extras else '') + '.'


def caption_bank(weights, tokenizer, rows_labels, rows_domains, annotated, names, device):
    """Unique captions of the training rows -> normalised CLIP text features; returns (features, key->index)."""
    from transformers import CLIPModel, CLIPTokenizer
    keys, texts = {}, []
    for y, d in zip(rows_labels, rows_domains):
        k = (int(d), np.packbits(y > 0.5).tobytes())
        if k not in keys:
            keys[k] = len(texts)
            texts.append(caption(y, annotated[d] > 0, names))
    model = CLIPModel.from_pretrained(weights, local_files_only=True).eval().to(device)
    tok = CLIPTokenizer.from_pretrained(tokenizer, local_files_only=True)
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            t = tok(texts[i:i + 512], padding=True, truncation=True, max_length=77, return_tensors='pt').to(device)
            feats.append(torch.nn.functional.normalize(model.get_text_features(**t).float(), dim=-1))
    del model
    print(f'caption bank: {len(texts)} unique captions, e.g. "{texts[0]}" | "{texts[len(texts) // 2]}"', flush=True)
    return torch.cat(feats), keys


def predict(model, loader, device):
    model.eval()
    probs, labels, domains = [], [], []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                enabled=device.type in ('cuda', 'mps')):
        for x, y, d in loader:
            logits = model(x.to(device, non_blocking=True)).float()
            if hasattr(model, 'domain_bias'):
                logits = logits + model.domain_bias[GENERIC]
            probs.append(logits.sigmoid().cpu().numpy())
            labels.append(y.numpy())
            domains.append(d.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.array(DOMAINS)[np.concatenate(domains)]


def wise(model, pretrained_state, finetuned_state, alpha):
    """WiSE-FT: interpolate between pre-trained and fine-tuned weights. A linear head
    has no meaningful pre-trained value, so only the text head is interpolated."""
    mix = lambda k, v: k in pretrained_state
    model.load_state_dict({k: (1 - alpha) * pretrained_state[k].to(v.device).float() + alpha * v.float()
                           if mix(k, v) else v for k, v in finetuned_state.items()}, strict=False)


def export(state, lora_scale=None):
    """Fold the generic domain-tag bias into the head bias (and LoRA updates into their weights) so the saved
    state is a plain PARModel."""
    state = dict(state)
    for k in [k for k in state if k.endswith('.lora_a')]:
        pre = k[:-len('lora_a')]
        a, b = state.pop(k), state.pop(pre + 'lora_b')
        state[pre + 'weight'] = state.pop(pre + 'base.weight') + lora_scale * b @ a
        if pre + 'base.bias' in state:
            state[pre + 'bias'] = state.pop(pre + 'base.bias')
    state.pop('crowd', None)  # per-source label transforms are training-only
    if 'domain_bias' in state:
        bias = state.pop('domain_bias')[GENERIC]
        key = 'head.net.1.bias' if 'head.net.1.bias' in state else 'head.bias'
        state[key] = state[key] + bias
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, help='CLIP tokenizer dir (default: --weights)')
    p.add_argument('--train-csv', type=Path, required=True)
    p.add_argument('--val-csv', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--held-out-domain', choices=DOMAINS, help='LODO fold; omit to train the final model on all domains')
    p.add_argument('--train-on-val', action='store_true', help='final model: train on train+val')
    p.add_argument('--size', type=int, nargs=2, default=(256, 128))
    p.add_argument('--aug', choices=('flip', 'strong'), default='strong')
    p.add_argument('--head', choices=('linear', 'text'), default='linear')
    p.add_argument('--extra-head', choices=('none', 'query', 'query1', 'query-nogate', 'region'), default='none',
                   help='patch-token head added to the CLS logits (zero-init)')
    p.add_argument('--heldout-every-epoch', action='store_true',
                   help='diagnostic: score the EMA on the held-out domain (prepare.py) after every epoch')
    p.add_argument('--teacher', nargs='+', help="B2: run dir of a frozen teacher (best_ema.pth + metrics.json); '{held}' is "
                   'replaced by the held-out domain, e.g. runs/lodo/B1l14/{held}')
    p.add_argument('--kd-weight', type=float, default=0.25, help='B2: weight of MSE(student logits, teacher logits)')
    p.add_argument('--fuse', type=int, nargs='*', default=(), help='F1: also use CLS of these blocks (1-based), zero-init head')
    p.add_argument('--attr-tokens', type=int, default=0, help='V1: 40 attribute tokens in the last k blocks')
    p.add_argument('--attr-text', type=Path, help='V1: full HF CLIP dir whose text tower initialises the tokens')
    p.add_argument('--pk', type=int, nargs=2, metavar=('P', 'K'),
                   help='Q1 stage 2: batches of P Market people x K images + P*K domain-balanced images')
    p.add_argument('--init-from', help='Q1 stage 2: checkpoint to start from ({held} = held-out domain)')
    p.add_argument('--ft-samples', type=int, default=0,
                   help='F1: set the number of fine-tuning epochs to round(ft_samples / training images)')
    p.add_argument('--wb-jitter', type=float, default=0.0, help='W1: per-channel gain jitter +-a (camera white balance)')
    p.add_argument('--gamma-jitter', type=float, default=0.0, help='W1: gamma = exp(U(-b, b)) (exposure)')
    p.add_argument('--exact-samples', action='store_true',
                   help='T1: with --ft-samples, stop after exactly that many images (truncate the last epoch)')
    p.add_argument('--id-proto', type=float, default=0.0,
                   help='P3: weight of logit MSE to the per-person running-mean prediction (Market ids)')
    p.add_argument('--id-consistency', type=float, default=0.0,
                   help='P1-id: weight of logit MSE between two images of the same Market person (agreeing labels only)')
    p.add_argument('--consistency', type=float, default=0.0,
                   help='C1: weight of MSE between logits of two differently augmented views (2x compute)')
    p.add_argument('--mixstyle', type=int, nargs='*', default=(), help='MixStyle after these block indices (1-based), p 0.5')
    p.add_argument('--llrd', type=float, default=1.0, help='layer-wise lr decay per block below the top (1 = off)')
    p.add_argument('--arch', choices=('b16', 'l14', 'siglip', 'pe'), default='b16', help='l14: CLIP ViT-L/14 (teacher experiments)')
    p.add_argument('--muon', action='store_true', help='Muon for the unfrozen 2-D backbone weights (AdamW for the rest)')
    p.add_argument('--softcap', type=float, default=0.0, help='logit soft-capping c; 0 = off')
    p.add_argument('--unfreeze', type=int, default=4, help='number of last blocks to train')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lp-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--backbone-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--freeze-top', type=int, default=0, help='S2: number of last blocks kept frozen')
    p.add_argument('--rev-llrd', action='store_true', help='S1: layer-wise lr decays from the first block upwards')
    p.add_argument('--lora', type=int, default=0, help='Q2: LoRA rank on all block Linears (0 = full fine-tune)')
    p.add_argument('--lora-alpha', type=float, default=0.0, help='Q2: LoRA alpha (default = rank)')
    p.add_argument('--feat-teacher', help='R1: dir of a frozen PE-Core checkpoint whose tokens the student matches')
    p.add_argument('--feat-kd', type=float, default=1.0, help='R1: weight of the token-level cosine distillation')
    p.add_argument('--elr', type=float, default=0.0, help='L2: ELR weight lambda (0 = off)')
    p.add_argument('--elr-beta', type=float, default=0.7, help='L2: ELR temporal-ensemble momentum')
    p.add_argument('--group-ce', type=float, default=0.0,
                   help='A1: weight of an auxiliary softmax CE over the upper/lower colour groups (one-colour images)')
    p.add_argument('--focal', type=float, default=0.0, help='G8: focal gamma on the BCE (0 = off)')
    p.add_argument('--pos-power', type=float, default=0.5)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--domain-balance', type=float, default=0.5)
    p.add_argument('--mask-missing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--min-positives', type=int, default=1,
                   help='A3: a (domain, attribute) pair with fewer positives counts as unannotated (masked)')
    p.add_argument('--caption-loss', type=float, default=0.0,
                   help='T1: weight of a multi-positive image-caption contrastive loss; captions built from the labels')
    p.add_argument('--logit-adjust', type=float, default=0.0,
                   help='A6: domain-aware logit adjustment (Menon et al. 2021): train on z + tau*logit(prior of the '
                        "image's source domain), predict with z; use with --pos-power 0")
    p.add_argument('--pw-effective', action='store_true',
                   help='A4: pos_weight from the sampler-weighted, annotation-masked training distribution')
    p.add_argument('--smooth-positives-only', action='store_true',
                   help='A2: label smoothing only lowers positive targets (1 -> 1-eps/2); negatives stay 0')
    p.add_argument('--wise-alphas', type=float, nargs='*', default=())
    p.add_argument('--crowd-colors', type=float, default=0.0, help='crowd-layer regulariser; 0 = off')
    p.add_argument('--lipsum', type=float, default=0.0,
                   help='L1 Lipsum-FT weight: MSE between fine-tuned and frozen image-text logits on random texts')
    p.add_argument('--lipsum-text', type=Path, help='full HF CLIP dir for the random-text features (default: --weights)')
    p.add_argument('--anchor', type=float, default=0.0, help='weight of reference anchoring; 0 = off')
    p.add_argument('--domain-tags', type=float, default=0.0, help='prob of generic tag; 0 = off')
    p.add_argument('--save-alpha', type=float, help='also save WiSE-FT weights at this alpha as final.pth')
    p.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    p.add_argument('--seed', type=int, default=605)
    p.add_argument('--drop-fold', type=int, nargs=2, metavar=('I', 'K'),
                   help='k-fold soup expert: leave out training fold I of K (split by image, per domain)')
    p.add_argument('--train-domains', nargs='+', choices=DOMAINS, help='domain expert: train on these only')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device {device}', flush=True)

    names, train_rows = read_rows(args.train_csv)
    _, val_rows = read_rows(args.val_csv)
    if args.limit:
        train_rows, val_rows = (random.sample(r, min(args.limit, len(r))) for r in (train_rows, val_rows))
    held = args.held_out_domain
    held_rows = [r for r in val_rows if held and r[0].startswith(held + '/')]
    train_rows = [r for r in train_rows if not (held and r[0].startswith(held + '/'))]
    val_rows = [r for r in val_rows if not (held and r[0].startswith(held + '/'))]
    if args.train_domains:  # selection still uses the validation images of all source domains
        train_rows = [r for r in train_rows if r[0].split('/')[0] in args.train_domains]
    if args.drop_fold:
        i, k = args.drop_fold
        order = np.random.default_rng(12345).permutation(len(train_rows))  # fixed split, independent of --seed
        train_rows = [train_rows[j] for n, j in enumerate(order) if n % k != i]
    if args.train_on_val:
        train_rows = train_rows + val_rows
    size = tuple(args.size)

    labels = np.asarray([r[1:] for r in train_rows], dtype=np.float32)
    domains = np.asarray([domain_of(r[0]) for r in train_rows])
    counts = np.bincount(domains, minlength=3).astype(float)
    sample_weight = counts[domains] ** -args.domain_balance
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_rows), replacement=True)
    positive = labels.sum(0)
    # (domain, attribute) pairs with no positive in training data were not annotated there
    annotated = np.stack([labels[domains == d].sum(0) >= args.min_positives if counts[d] else np.ones(40, bool)
                          for d in range(3)]).astype(np.float32)
    if not args.mask_missing:
        annotated[:] = 1
    if args.pw_effective:  # A4: neg/pos under the distribution the loss actually sees (sampler weights, masked pairs)
        w = sample_weight[:, None] * annotated[domains]
        eff_pos, eff_all = (w * labels).sum(0), w.sum(0)
        pos_weight = ((eff_all - eff_pos) / np.maximum(eff_pos, 1e-8 + eff_all * 1e-6)) ** args.pos_power
    else:
        pos_weight = ((len(labels) - positive) / np.maximum(positive, 1)) ** args.pos_power

    # spawn: workers do not inherit the parent's CPU copies of the CLIP weights (shared server RAM)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == 'cuda',
                     multiprocessing_context='spawn' if args.workers else None)
    train_set = Images(train_rows, args.image_root, size, True, args.aug)
    train_set.wb, train_set.gamma = args.wb_jitter, args.gamma_jitter
    if args.elr:  # L2: early-learning regularisation (Liu et al., NeurIPS 2020), binary per attribute
        train_set.with_index = True
        elr_t = torch.zeros(len(train_rows), 40, device=device)
        elr_seen = torch.zeros(len(train_rows), dtype=torch.bool, device=device)
    if args.id_proto or args.pk:
        n_people = train_set.set_identity_index()
        proto = torch.zeros(n_people, 40, device=device)
        seen = torch.zeros(n_people, dtype=torch.bool, device=device)
        print(f'identity prototypes: {n_people} people', flush=True)
    if args.id_consistency:
        print(f'identity pairs: {train_set.set_identity_pairs()} images with a same-person partner', flush=True)
    if args.pk:  # Q1 stage 2: person-grouped batches
        train_loader = DataLoader(train_set, batch_sampler=PKBatches(train_set.person, sample_weight, *args.pk),
                                  persistent_workers=args.workers > 0, **loader_kw)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size, sampler=sampler, drop_last=True,
                                  persistent_workers=args.workers > 0, **loader_kw)
    # val loader keeps its workers: spawned once, not every epoch (worker segfaults at spawn hang the run)
    eval_loader = lambda rows, keep=False: DataLoader(Images(rows, args.image_root, size), batch_size=256,
                                                      persistent_workers=keep and args.workers > 0, **loader_kw)
    val_loader = eval_loader(val_rows, True) if val_rows and not args.train_on_val else None
    held_loader = eval_loader(held_rows) if held_rows else None

    model = PARPlus(size, args.head, args.extra_head, names, args.softcap, args.arch)
    model.mixstyle_layers = tuple(args.mixstyle)
    if args.fuse:
        model.set_fuse(args.fuse)
    weights_file = args.weights if args.weights.is_file() else args.weights / 'pytorch_model.bin'
    state = torch.load(weights_file, map_location='cpu', weights_only=True)  # HF CLIP or our shared.pth
    state = {k: v.float() for k, v in state.items()}
    model.vision.load_hf_state(state)
    prior = np.clip(positive / len(labels), 1e-4, 1 - 1e-4)
    with torch.no_grad():
        if args.head == 'text':
            model.head.text.copy_(text_embeddings(args.weights, args.tokenizer or args.weights, names))
            # bias so that the zero-shot mean similarity maps to the class prior
            model.head.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32)
                                  - model.head.logit_scale.exp() * 0.2)
        else:
            model.head.net[1].bias.copy_(torch.tensor(np.log(prior / (1 - prior)) * (1 - args.logit_adjust)))
    del state
    lora_scale = None
    if args.lora:  # Q2: low-distortion fine-tuning; merged before checkpoints are written
        add_lora(model.vision.layers, args.lora, args.lora_alpha or args.lora)
        lora_scale = (args.lora_alpha or args.lora) / args.lora
    if args.init_from:  # Q1 stage 2: continue from a finished run of this fold
        init = Path(str(args.init_from).format(held=held))
        model.load_state_dict({k: v.float() for k, v in torch.load(init, map_location='cpu', weights_only=True).items()},
                              strict=False)
        print(f'initialised from {init}', flush=True)
    if args.attr_tokens:
        text = text_embeddings(args.attr_text, args.tokenizer or args.attr_text, names) if args.attr_text else None
        model.set_attr_tokens(args.attr_tokens, text)
    if args.domain_tags:
        model.domain_bias = torch.nn.Parameter(torch.zeros(GENERIC + 1, 40))
    colour_groups = [[i for i, n in enumerate(names) if n.startswith(part + '-Color')]
                     for part in ('UpperBody', 'LowerBody')]
    if args.crowd_colors:
        model.crowd = torch.nn.Parameter(torch.zeros(len(DOMAINS), 2, 12, 12))  # delta from identity
    keep = lambda k: args.head == 'text' or k.startswith('vision.')
    pretrained_state = {k: v.detach().clone().half() for k, v in model.state_dict().items()
                        if v.is_floating_point() and keep(k)}
    model.to(device)
    first = len(model.vision.layers) - args.unfreeze
    reference = copy.deepcopy(model.vision).eval().requires_grad_(False) if args.anchor or args.lipsum else None
    bank = lipsum_bank(args.lipsum_text or args.weights).to(device) if args.lipsum else None
    model.vision.requires_grad_(False)
    top = [model.vision.post_layernorm, model.vision.visual_projection, *([model.vision.head] if hasattr(model.vision, 'head') else [])]
    tuned = list(model.vision.layers[-args.unfreeze:])
    if args.freeze_top:  # S2: keep the last blocks at their pretrained weights (surgical fine-tuning)
        tuned = tuned[:-args.freeze_top]
    blocks = [*tuned, *top]
    lora_blocks = set(map(id, model.vision.layers)) if args.lora else set()
    backbone_params = [q for b in blocks for n, q in b.named_parameters() if id(b) not in lora_blocks or 'lora_' in n]
    feat_teacher = None
    if args.feat_teacher:  # R1 (AM-RADIO-style): match every token of a frozen foundation model through an adaptor
        feat_teacher = TimmPEVision(size)
        feat_teacher.load_hf_state(torch.load(Path(args.feat_teacher) / 'pytorch_model.bin', map_location='cpu', weights_only=True))
        feat_teacher = feat_teacher.to(device).eval().requires_grad_(False)
        width = model.vision.post_layernorm.normalized_shape[0]
        model.adaptor = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, 1024), torch.nn.GELU(),
                                            torch.nn.Linear(1024, 768)).to(device)
        print(f'feature teacher {args.feat_teacher}', flush=True)
    head_params = (list(model.head.parameters()) + (list(model.extra.parameters()) if model.extra else [])
                   + (list(model.fuse.parameters()) if model.fuse is not None else [])
                   + ([model.attr_tokens, model.attr_w, model.attr_b] if args.attr_tokens else [])
                   + (list(model.adaptor.parameters()) if feat_teacher is not None else []) + ([model.domain_bias] if args.domain_tags else [])
                   + ([model.crowd] if args.crowd_colors else []))
    muon_params = [q for q in backbone_params if q.ndim == 2] if args.muon else []
    adam_backbone = [q for q in backbone_params if not any(q is m for m in muon_params)]
    if args.llrd < 1.0:  # layer-wise lr decay: top unfrozen block gets backbone_lr, each lower one x llrd
        depth = {id(q): 0 for b in top for q in b.parameters()}
        order = tuned if args.rev_llrd else list(reversed(tuned))  # S1: reverse = lowest block gets the full lr
        for i, layer in enumerate(order):
            depth.update({id(q): i for q in layer.parameters()})
        groups = {}
        for q in adam_backbone:
            groups.setdefault(depth[id(q)], []).append(q)
        backbone_groups = [{'params': g, 'lr': args.backbone_lr * args.llrd ** k, 'weight_decay': args.weight_decay}
                           for k, g in sorted(groups.items())]
    else:
        backbone_groups = [{'params': adam_backbone, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.AdamW(backbone_groups + [{'params': head_params, 'lr': args.head_lr, 'weight_decay': 0.0}])
    muon = Muon(muon_params, lr=args.backbone_lr, weight_decay=args.weight_decay) if muon_params else None
    if args.ft_samples:  # F1: fine-tuning length in images seen, not epochs (held-out peaks at ~180k images)
        args.epochs = max(1, round(args.ft_samples / len(train_rows)))
        print(f'ft-samples {args.ft_samples} -> {args.epochs} fine-tuning epochs of {len(train_rows)} images', flush=True)
    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    ft_steps = None
    if args.ft_samples and args.exact_samples:  # T1: stop after exactly ft_samples images (last epoch truncated)
        ft_steps = max(1, args.ft_samples // args.batch_size)
        args.epochs, total = math.ceil(ft_steps / steps_per_epoch), ft_steps
        print(f'exact: {ft_steps} fine-tuning steps over {args.epochs} epochs', flush=True)
    ft_done = 0
    warmup = steps_per_epoch // 2

    def lr_factor(step):  # LP epochs use the head lr only; backbone gets warmup + cosine
        s = step - args.lp_epochs * steps_per_epoch
        if s < 0:
            return 1.0
        return min(1.0, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * s / max(total, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    muon_scheduler = torch.optim.lr_scheduler.LambdaLR(muon, lr_factor) if muon else None
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    annotated_t = torch.tensor(annotated, device=device)
    teachers = []  # B2/K1: frozen teachers see the same augmented batch (resized to their input size); logits averaged
    for tpath in args.teacher or []:
        tdir = Path(tpath.format(held=held))
        tmeta = json.loads((tdir / 'metrics.json').read_text())
        tsize = tuple(tmeta['size'])
        teacher = PARPlus(tsize, tmeta['head'], arch=tmeta.get('arch', 'b16'))
        tw = Path(tmeta['weights'])
        tw = tw if tw.is_file() else tw / 'pytorch_model.bin'
        teacher.vision.load_hf_state({k: v.float() for k, v in torch.load(tw, map_location='cpu', weights_only=True).items()})
        teacher.load_state_dict({k: v.float() for k, v in torch.load(tdir / 'best_ema.pth', map_location='cpu',
                                                                   weights_only=True).items()}, strict=False)
        teachers.append((teacher.to(device).eval().requires_grad_(False), tsize))
        print(f'teacher {tdir} arch {tmeta.get("arch", "b16")} size {tsize}', flush=True)
    if args.caption_loss:
        cap_feats, cap_keys = caption_bank(args.weights, args.tokenizer or args.weights, labels, domains,
                                           annotated, names, device)
    # per-domain prior (pooled prior where the pair is not annotated) -> tau * logit offset per training domain
    dom_prior = np.stack([np.where(annotated[k] > 0, labels[domains == k].mean(0) if counts[k] else prior, prior)
                          for k in range(3)])
    dom_prior = np.clip(dom_prior, 1e-4, 1 - 1e-4)
    la_offset = torch.tensor(args.logit_adjust * np.log(dom_prior / (1 - dom_prior)), dtype=torch.float32, device=device)
    eps = args.label_smoothing

    history, best, best_epoch = [], -1.0, 0
    ema_bias_snapshot = None
    started = time.monotonic()
    for epoch in range(1, args.lp_epochs + args.epochs + 1):
        lp = epoch <= args.lp_epochs
        for q in backbone_params:
            q.requires_grad_(not lp)
        model.train()
        t0, loss_sum, seen_n = time.monotonic(), 0.0, 0
        for batch in train_loader:
            x, y, d = batch[:3]
            x, y, d = x.to(device, non_blocking=True), y.to(device, non_blocking=True), d.to(device)
            anchor = lipsum = torch.zeros((), device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ('cuda', 'mps')):
                if reference is None or lp:
                    logits = model(x).float()
                else:
                    with torch.no_grad():  # layers below `first` are frozen and shared
                        shared = model.vision.tokens(x, first)
                        _, ref_embed = reference.tail(shared, first)
                    pooled, embed = model.vision.tail(shared, first)
                    logits = model.head(pooled, embed).float()
                    anchor = 1 - torch.nn.functional.cosine_similarity(
                        embed.float(), ref_embed.float(), dim=-1).mean()
                    if bank is not None:
                        t = bank[torch.randint(len(bank), (80,), device=device)]
                        energy = lambda e: 100 * torch.nn.functional.normalize(e.float(), dim=-1) @ t.T
                        lipsum = (energy(embed) - energy(ref_embed)).pow(2).mean()
            if args.domain_tags:
                tag = torch.where(torch.rand(len(d), device=device) < args.domain_tags,
                                  torch.full_like(d, GENERIC), d)
                logits = logits + model.domain_bias[tag]
            if args.crowd_colors:
                noisy = logits.clone()
                for g, idx in enumerate(colour_groups):
                    clean = logits[:, idx]
                    noisy[:, idx] = clean + torch.einsum('bi,bij->bj', clean, model.crowd[d, g])
                logits = noisy
            if args.logit_adjust:
                logits = logits + la_offset[d]
            target = y * (1 - eps / 2) if args.smooth_positives_only else y * (1 - eps) + eps / 2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight_t, reduction='none')
            if args.focal:  # G8: focal modulation (1 - p_t)^gamma on top of pos_weight (hard, e.g. occluded, examples)
                p_t = torch.where(y > 0.5, torch.sigmoid(logits), torch.sigmoid(-logits))
                loss = loss * (1 - p_t).detach().pow(args.focal)
            weight = annotated_t[d]
            loss = (loss * weight).sum() / weight.sum() + args.anchor * anchor + args.lipsum * lipsum
            if args.consistency and not lp:  # C1: second view = flip + brightness/contrast + random erasing (GPU)
                x2 = x.flip(-1)
                g = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=device) - 0.5)
                x2 = (x2 - x2.mean((2, 3), keepdim=True)) * g + x2.mean((2, 3), keepdim=True) + 0.2 * (torch.rand(len(x), 1, 1, 1, device=device) - 0.5)
                h, w = x2.shape[-2:]
                eh, ew = h // 4, w // 4
                y0, x0 = int(torch.randint(0, h - eh, ())), int(torch.randint(0, w - ew, ()))
                x2[:, :, y0:y0 + eh, x0:x0 + ew] = 0
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ('cuda', 'mps')):
                    logits2 = model(x2).float()
                loss = loss + args.consistency * (logits - logits2).pow(2).mean()
            if args.id_consistency and not lp and batch[5].any():  # P1-id: same person, other image
                has = batch[5].to(device)
                x2, y2 = batch[3].to(device, non_blocking=True)[has], batch[4].to(device)[has]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ('cuda', 'mps')):
                    logits2 = model(x2).float()
                agree = (y[has] == y2).float() * annotated_t[d[has]]  # only where both images carry the same label
                loss = loss + args.id_consistency * ((logits[has] - logits2).pow(2) * agree).sum() / agree.sum().clamp(min=1)
            if args.id_proto and not lp:  # P3: pull each image towards its person's mean prediction (EMA bank)
                pid = batch[3].to(device)
                m = pid >= 0
                if m.any():
                    pm = pid[m]
                    with torch.no_grad():
                        cur = logits[m].detach()
                        proto[pm] = torch.where(seen[pm, None], 0.9 * proto[pm] + 0.1 * cur, cur)
                        seen[pm] = True
                    loss = loss + args.id_proto * (logits[m] - proto[pm]).pow(2).mean()
            if args.elr and not lp:  # pull towards the temporal ensemble of this image's own earlier predictions
                ii = batch[3].to(device)
                prob = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
                with torch.no_grad():
                    cur = prob.detach()
                    elr_t[ii] = torch.where(elr_seen[ii, None], args.elr_beta * elr_t[ii] + (1 - args.elr_beta) * cur, cur)
                    elr_seen[ii] = True
                agree = prob * elr_t[ii] + (1 - prob) * (1 - elr_t[ii])
                loss = loss + args.elr * torch.log(1 - agree).mean()
            if args.group_ce and not lp:  # A1: colour groups are ~categorical (98.8-99.8 % one colour): auxiliary
                for idx in colour_groups:      # softmax CE on images with exactly one annotated colour in the group
                    yg = y[:, idx]
                    one = (yg.sum(1) == 1) & (annotated_t[d][:, idx].min(1).values > 0)
                    if one.any():
                        loss = loss + args.group_ce * torch.nn.functional.cross_entropy(logits[one][:, idx], yg[one].argmax(1))
            if feat_teacher is not None and not lp:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    tt = feat_teacher.post_layernorm(feat_teacher.tokens(x)).float()
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    st = model.adaptor(model.last_tokens).float()
                n = min(st.shape[1], tt.shape[1])  # same grid; CLS + patches
                loss = loss + args.feat_kd * (1 - torch.nn.functional.cosine_similarity(st[:, -n:], tt[:, -n:], dim=-1)).mean()
            if teachers:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    t_logits = sum(t(x if ts == size else torch.nn.functional.interpolate(
                        x, size=ts, mode='bilinear', align_corners=False)).float() for t, ts in teachers) / len(teachers)
                loss = loss + args.kd_weight * (logits - t_logits).pow(2).mean()
            if args.caption_loss and not lp:
                yb, db = y.cpu().numpy(), d.cpu().numpy()
                ids = torch.tensor([cap_keys[(int(k), np.packbits(v > 0.5).tobytes())] for v, k in zip(yb, db)],
                                   device=device)
                img = torch.nn.functional.normalize(model.last_embed.float(), dim=-1)
                txt = cap_feats[ids]
                same = (ids[:, None] == ids[None, :]).float()  # identical captions are all positives
                sim = 30.0 * img @ txt.t()
                loss = loss + args.caption_loss * -(same / same.sum(1, keepdim=True) * sim.log_softmax(1)).sum(1).mean()
            if args.crowd_colors:
                loss = loss + args.crowd_colors * model.crowd.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            if muon:
                muon.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if muon:
                muon.step()
                muon_scheduler.step()
            loss_sum += float(loss.detach()) * len(x)
            seen_n += len(x)
            decay = 0.0 if lp else args.ema  # EMA only tracks the fine-tuning phase
            with torch.no_grad():
                for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                    e.copy_(m) if decay == 0.0 or not e.is_floating_point() else e.lerp_(m, 1 - decay)
            ft_done += not lp
            if ft_steps is not None and ft_done >= ft_steps:
                break
        row = {'epoch': epoch, 'phase': 'lp' if lp else 'ft', 'train_loss': loss_sum / max(seen_n, 1),
               'train_seconds': time.monotonic() - t0}
        if val_loader is not None:
            probs, truth, doms = predict(ema, val_loader, device)
            metrics = challenge_score(truth, probs, doms)
            row.update({k: metrics[k] for k in ('mA', 'Inst_F1', 'Challenge_Avg')})
            if metrics['Challenge_Avg'] > best and not lp:
                best, best_epoch = metrics['Challenge_Avg'], epoch
                torch.save(export(ema.state_dict(), lora_scale), args.output_dir / 'best_ema.pth')
        if args.domain_tags:
            ema_bias_snapshot = ema.domain_bias.detach().cpu().tolist()
        if args.heldout_every_epoch:  # diagnostic only; selection still uses source validation
            h = prepare.evaluate_heldout(ema, size, held, prior, device, args.workers)
            row.update(heldout_cc=h['cc']['Challenge_Avg'], heldout_raw=h['raw']['Challenge_Avg'])
        history.append(row)
        print(json.dumps(row), flush=True)
    if val_loader is None or best_epoch == 0:  # no selection split: keep the final EMA
        torch.save(export(ema.state_dict(), lora_scale), args.output_dir / 'best_ema.pth')
        best_epoch = history[-1]['epoch']

    result = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'train_images': len(train_rows), 'best_epoch': best_epoch, 'source_val_best': best,
              'pos_weight_max': float(pos_weight.max()), 'annotated': annotated.tolist(),
              'total_seconds': time.monotonic() - started, 'history': history}
    finetuned = torch.load(args.output_dir / 'best_ema.pth', map_location=device, weights_only=True)
    if hasattr(ema, 'domain_bias'):
        ema.domain_bias.data.zero_()  # already folded into the head bias by export()
        result['domain_bias'] = ema_bias_snapshot
    if args.lora:
        merge_lora(ema)  # the saved best checkpoint holds merged weights
    missing = ema.load_state_dict(finetuned, strict=False).missing_keys
    assert not [k for k in missing if k.startswith('vision.')], missing[:5]
    if held:
        result['prepare'] = prepare.evaluate_heldout(ema, size, held, prior, device, args.workers,
                                                     save=args.output_dir / 'probe.npz')
    if held_loader is not None and args.wise_alphas:
        result['heldout'] = {}
        for alpha in args.wise_alphas:
            wise(ema, pretrained_state, finetuned, alpha)
            probs, truth, doms = predict(ema, held_loader, device)
            metrics = challenge_score(truth, probs, doms)
            metrics.pop('domains')
            result['heldout'][str(alpha)] = metrics
            np.save(args.output_dir / f'heldout_probs_a{alpha}.npy', probs)
            print(f'held-out {held} alpha={alpha}: {json.dumps(metrics)}', flush=True)
        np.save(args.output_dir / 'heldout_labels.npy', truth)
    if args.save_alpha is not None:
        wise(ema, pretrained_state, finetuned, args.save_alpha)
        torch.save(export(ema.state_dict(), lora_scale), args.output_dir / 'final.pth')
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    if held:
        prepare.print_summary(held, result['prepare'], result['total_seconds'])


if __name__ == '__main__':
    main()
