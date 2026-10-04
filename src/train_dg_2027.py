"""Domain-generalisation fine-tuning of CLIP ViT-B/16 for UPAR 2027 Track 1.

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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
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
        x = (x - self.mean) / self.std
        if strong and random.random() < 0.3:  # random erasing
            eh, ew = random.randint(h // 8, h // 3), random.randint(w // 8, w // 3)
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            x[:, y0:y0 + eh, x0:x0 + ew] = torch.randn(3, eh, ew)
        return x, self.labels[index], domain_of(path)


DOMAINS = ('Market1501', 'PA100k', 'PETA')
GENERIC = len(DOMAINS)  # index of the domain-agnostic tag


def domain_of(path):
    return DOMAINS.index(path.split('/')[0])


def text_embeddings(weights, tokenizer, names):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained(weights, local_files_only=True).eval()
    tokens = CLIPTokenizer.from_pretrained(tokenizer, local_files_only=True)(
        [prompt(n) for n in names], padding=True, return_tensors='pt')
    with torch.no_grad():
        return model.get_text_features(**tokens)


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


def export(state):
    """Fold the generic domain-tag bias into the head bias so the saved state is a plain PARModel."""
    state = dict(state)
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
    p.add_argument('--held-out-domain', choices=DOMAINS)
    p.add_argument('--train-on-val', action='store_true', help='final model: train on train+val')
    p.add_argument('--size', type=int, nargs=2, default=(256, 128))
    p.add_argument('--aug', choices=('flip', 'strong'), default='strong')
    p.add_argument('--head', choices=('linear', 'text'), default='linear')
    p.add_argument('--unfreeze', type=int, default=2, help='number of last blocks to train')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lp-epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--backbone-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--pos-power', type=float, default=0.5)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--domain-balance', type=float, default=0.5)
    p.add_argument('--mask-missing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--wise-alphas', type=float, nargs='*', default=(0.5, 0.7, 1.0))
    p.add_argument('--crowd-colors', type=float, default=0.0, help='crowd-layer regulariser; 0 = off')
    p.add_argument('--anchor', type=float, default=0.0, help='weight of reference anchoring; 0 = off')
    p.add_argument('--domain-tags', type=float, default=0.0, help='prob of generic tag; 0 = off')
    p.add_argument('--save-alpha', type=float, help='also save WiSE-FT weights at this alpha as final.pth')
    p.add_argument('--limit', type=int, default=0, help='smoke test: rows per split')
    p.add_argument('--seed', type=int, default=605)
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
    if args.train_on_val:
        train_rows = train_rows + val_rows
    size = tuple(args.size)

    labels = np.asarray([r[1:] for r in train_rows], dtype=np.float32)
    domains = np.asarray([domain_of(r[0]) for r in train_rows])
    counts = np.bincount(domains, minlength=3).astype(float)
    sample_weight = counts[domains] ** -args.domain_balance
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(train_rows), replacement=True)
    positive = labels.sum(0)
    pos_weight = ((len(labels) - positive) / np.maximum(positive, 1)) ** args.pos_power
    # (domain, attribute) pairs with no positive in training data were not annotated there
    annotated = np.stack([labels[domains == d].sum(0) > 0 if counts[d] else np.ones(40, bool)
                          for d in range(3)]).astype(np.float32)
    if not args.mask_missing:
        annotated[:] = 1

    # spawn: workers do not inherit the parent's CPU copies of the CLIP weights (shared server RAM)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == 'cuda',
                     multiprocessing_context='spawn' if args.workers else None)
    train_loader = DataLoader(Images(train_rows, args.image_root, size, True, args.aug),
                              batch_size=args.batch_size, sampler=sampler, drop_last=True,
                              persistent_workers=args.workers > 0, **loader_kw)
    eval_loader = lambda rows: DataLoader(Images(rows, args.image_root, size), batch_size=256, **loader_kw)
    val_loader = eval_loader(val_rows) if val_rows and not args.train_on_val else None
    held_loader = eval_loader(held_rows) if held_rows else None

    model = PARModel(size, args.head)
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
            model.head.net[1].bias.copy_(torch.tensor(np.log(prior / (1 - prior))))
    del state
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
    reference = copy.deepcopy(model.vision).eval().requires_grad_(False) if args.anchor else None
    model.vision.requires_grad_(False)
    blocks = [*model.vision.layers[-args.unfreeze:], model.vision.post_layernorm, model.vision.visual_projection]
    backbone_params = [q for b in blocks for q in b.parameters()]
    head_params = (list(model.head.parameters()) + ([model.domain_bias] if args.domain_tags else [])
                   + ([model.crowd] if args.crowd_colors else []))
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
        {'params': head_params, 'lr': args.head_lr, 'weight_decay': 0.0}])
    steps_per_epoch = len(train_loader)
    total = args.epochs * steps_per_epoch
    warmup = steps_per_epoch // 2

    def lr_factor(step):  # LP epochs use the head lr only; backbone gets warmup + cosine
        s = step - args.lp_epochs * steps_per_epoch
        if s < 0:
            return 1.0
        return min(1.0, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * s / max(total, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    annotated_t = torch.tensor(annotated, device=device)
    eps = args.label_smoothing

    history, best, best_epoch = [], -1.0, 0
    ema_bias_snapshot = None
    started = time.monotonic()
    for epoch in range(1, args.lp_epochs + args.epochs + 1):
        lp = epoch <= args.lp_epochs
        for q in backbone_params:
            q.requires_grad_(not lp)
        model.train()
        t0, loss_sum = time.monotonic(), 0.0
        for x, y, d in train_loader:
            x, y, d = x.to(device, non_blocking=True), y.to(device, non_blocking=True), d.to(device)
            anchor = torch.zeros((), device=device)
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
            target = y * (1 - eps) + eps / 2
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight_t, reduction='none')
            weight = annotated_t[d]
            loss = (loss * weight).sum() / weight.sum() + args.anchor * anchor
            if args.crowd_colors:
                loss = loss + args.crowd_colors * model.crowd.pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(x)
            decay = 0.0 if lp else args.ema  # EMA only tracks the fine-tuning phase
            with torch.no_grad():
                for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                    e.copy_(m) if decay == 0.0 or not e.is_floating_point() else e.lerp_(m, 1 - decay)
        row = {'epoch': epoch, 'phase': 'lp' if lp else 'ft', 'train_loss': loss_sum / len(train_loader.sampler),
               'train_seconds': time.monotonic() - t0}
        if val_loader is not None:
            probs, truth, doms = predict(ema, val_loader, device)
            metrics = challenge_score(truth, probs, doms)
            row.update({k: metrics[k] for k in ('mA', 'Inst_F1', 'Challenge_Avg')})
            if metrics['Challenge_Avg'] > best and not lp:
                best, best_epoch = metrics['Challenge_Avg'], epoch
                torch.save(export(ema.state_dict()), args.output_dir / 'best_ema.pth')
        if args.domain_tags:
            ema_bias_snapshot = ema.domain_bias.detach().cpu().tolist()
        history.append(row)
        print(json.dumps(row), flush=True)
    if val_loader is None or best_epoch == 0:  # no selection split: keep the final EMA
        torch.save(export(ema.state_dict()), args.output_dir / 'best_ema.pth')
        best_epoch = history[-1]['epoch']

    result = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              'train_images': len(train_rows), 'best_epoch': best_epoch, 'source_val_best': best,
              'pos_weight_max': float(pos_weight.max()), 'annotated': annotated.tolist(),
              'total_seconds': time.monotonic() - started, 'history': history}
    finetuned = torch.load(args.output_dir / 'best_ema.pth', map_location=device, weights_only=True)
    if hasattr(ema, 'domain_bias'):
        ema.domain_bias.data.zero_()  # already folded into the head bias by export()
        result['domain_bias'] = ema_bias_snapshot
    if held_loader is not None:
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
        torch.save(export(ema.state_dict()), args.output_dir / 'final.pth')
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
