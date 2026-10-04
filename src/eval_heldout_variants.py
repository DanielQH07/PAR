"""Inference-only probes on a LODO checkpoint's held-out domain:
  * logits on original and h-flipped images (self-consistency / TTA),
  * logits on content-free inputs (mean canvas, black, white, blurred noise) for
    contextual calibration in the spirit of "Calibrate Before Use" (Zhao et al. 2021).
Saves logits so combinations can be scored offline without re-running the model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clip_vit import CLIP_MEAN, CLIP_STD, PARModel
from train_dg_2027 import Images, read_rows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_submission import legacy_to_parmodel  # noqa: E402


def load(kind, path, weights):
    if kind == 'dg':
        run = json.loads((path / 'metrics.json').read_text())
        model = PARModel(tuple(run['size']), run['head'])
        state = torch.load(path / 'best_ema.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        return model, tuple(run['size']), run['held_out_domain']
    model = PARModel((224, 224), 'linear')
    model.vision.load_hf_state(torch.load(weights / 'pytorch_model.bin', map_location='cpu', weights_only=True))
    own = model.state_dict()
    state = legacy_to_parmodel(torch.load(path / 'best_model.pth', map_location='cpu', weights_only=True))
    state = {k: v for k, v in state.items() if k in own}  # drops HF position_ids buffer
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith('vision.') for k in missing), missing
    held = json.loads((path / 'metrics.json').read_text())['held_out_domain']
    return model, (224, 224), held


def content_free(size):
    h, w = size
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    norm = lambda x: (x - mean) / std
    g = torch.Generator().manual_seed(0)
    noise = torch.rand(3, h // 8, w // 8, generator=g)
    noise = torch.nn.functional.interpolate(noise[None], size=(h, w), mode='bilinear')[0]
    return torch.stack([torch.zeros(3, h, w), norm(torch.zeros(3, h, w)),
                        norm(torch.ones(3, h, w)), norm(noise)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--member', required=True, help='dg:<run dir> or legacy:<run dir>')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--val-csv', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    kind, path = args.member.split(':', 1)
    model, size, held = load(kind, Path(path), args.weights)
    device = torch.device('cuda')
    model.eval().to(device)
    _, rows = read_rows(args.val_csv)
    rows = [r for r in rows if r[0].startswith(held + '/')]
    loader = DataLoader(Images(rows, args.image_root, size), batch_size=256, num_workers=4,
                        multiprocessing_context='spawn')
    logits, flipped, labels = [], [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x, y, _ in loader:
            x = x.to(device)
            logits.append(model(x).float().cpu())
            flipped.append(model(x.flip(-1)).float().cpu())
            labels.append(y)
        cf = model(content_free(size).to(device)).float().cpu()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'probe.npz', logits=torch.cat(logits).numpy(),
             flipped=torch.cat(flipped).numpy(), labels=torch.cat(labels).numpy(),
             content_free=cf.numpy(), held=held)
    print(held, len(rows), 'images ->', args.output / 'probe.npz')


if __name__ == '__main__':
    main()
