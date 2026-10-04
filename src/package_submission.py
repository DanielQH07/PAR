"""Build the Codabench zip for an ensemble of CLIP ViT-B/16 members.

Each --member is  kind:path[:checkpoint]
  dg:runs/full_linear[:best_ema.pth]   a train_dg_2027.py run (size/head read from metrics.json)
  legacy:.../best_model.pth            a train_partial_ft_2027.py CLIP checkpoint (224x224 pad)
The frozen pre-trained CLIP weights are stored once (shared.pth, fp16); each member
keeps only the trainable tail (last blocks, post-LN, projection) and its head.

--cc gives one contextual-calibration strength per member ("Calibrate Before Use",
Zhao et al. 2021): correction = lambda * (mean logit on content-free images -
logit of the training prior), subtracted from the member's logits at test time.
--tta averages logits over the image and its horizontal flip.
"""

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def tail_only(state, blocks):
    """Keep head + last `blocks` encoder layers + post-LN + projection."""
    first = 12 - blocks
    keep = lambda k: (k.startswith('head.') or k.startswith('vision.post_layernorm')
                      or k.startswith('vision.visual_projection')
                      or (k.startswith('vision.layers.') and int(k.split('.')[2]) >= first))
    return {k: v.half() for k, v in state.items() if keep(k)}


def legacy_to_parmodel(checkpoint):
    """train_partial_ft_2027.py stores HF CLIPVisionTransformer names + Sequential head."""
    out = {}
    for k, v in checkpoint['vision'].items():
        k = k.replace('encoder.layers.', 'layers.').replace('.self_attn.', '.').replace('.mlp.', '.')
        out['vision.' + k] = v
    for k, v in checkpoint['head'].items():
        out['head.net.' + k] = v
    return out


def calibration(state, size, head, weights, prior, lam):
    """Precompute the contextual-calibration offset for one member (fp32, CPU)."""
    if not lam:
        return [0.0] * 40
    from clip_vit import PARModel
    from eval_heldout_variants import content_free
    model = PARModel(tuple(size), head)
    model.vision.load_hf_state(torch.load(weights / 'pytorch_model.bin', map_location='cpu',
                                          weights_only=True))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not unexpected and all(k.startswith('vision.') for k in missing)
    with torch.no_grad():
        cf = model.eval()(content_free(tuple(size))).mean(0).numpy()
    return (lam * (cf - np.log(prior / (1 - prior)))).astype(float).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--member', action='append', required=True)
    p.add_argument('--weights', type=Path, required=True, help='HF CLIP dir with pytorch_model.bin')
    p.add_argument('--val-csv', type=Path, required=True, help='for the attribute order')
    p.add_argument('--out', type=Path, required=True, help='build dir; <out>.zip is written too')
    p.add_argument('--cc', type=float, nargs='*', default=(), help='lambda per member (default 0)')
    p.add_argument('--train-csv', type=Path, required=True, help='training labels (prior)')
    p.add_argument('--tta', action='store_true', help='h-flip test-time augmentation')
    p.add_argument('--cpu-size', type=int, nargs=2, help='input size of member 0 on CPU-only workers')
    p.add_argument('--cpu-cc', type=float, help='calibration lambda of member 0 on CPU-only workers')
    args = p.parse_args()
    with args.val_csv.open(newline='') as stream:
        names = next(csv.reader(stream))[1:]
    assert len(names) == 40
    args.out.mkdir(parents=True, exist_ok=False)
    template = ROOT / 'submission' / 'task1'
    for name in ('run.py', 'metadata.yaml'):
        shutil.copy(template / name, args.out / name)
    shutil.copy(ROOT / 'src' / 'clip_vit.py', args.out / 'clip_vit.py')
    pretrained = torch.load(args.weights / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    torch.save({k: v.half() for k, v in pretrained.items()
                if k.startswith('vision_model.') or k == 'visual_projection.weight'},
               args.out / 'shared.pth')
    lams = list(args.cc) + [0.0] * (len(args.member) - len(args.cc))
    with args.train_csv.open(newline='') as stream:  # label prior: calibration + deadline fallback
        rows = list(csv.reader(stream))[1:]
    prior = np.clip(np.asarray([r[1:] for r in rows], dtype=np.float64).mean(0), 1e-4, 1 - 1e-4)
    members = []
    for i, spec in enumerate(args.member):
        kind, path, *rest = spec.split(':')
        path = Path(path)
        if kind == 'dg':
            run = json.loads((path / 'metrics.json').read_text())
            state = torch.load(path / (rest[0] if rest else 'best_ema.pth'), map_location='cpu',
                               weights_only=True)
            size, head, blocks = run['size'], run['head'], run['unfreeze']
        elif kind == 'legacy':
            state = legacy_to_parmodel(torch.load(path, map_location='cpu', weights_only=True))
            size, head, blocks = [224, 224], 'linear', 2
        else:
            raise ValueError(spec)
        file = f'member_{i}.pth'
        tail = tail_only(state, blocks)
        torch.save(tail, args.out / file)
        member = {'file': file, 'size': list(size), 'head': head, 'source': spec, 'cc_lambda': lams[i],
                  'correction': calibration(tail, size, head, args.weights, prior, lams[i])}
        if i == 0 and args.cpu_size:  # run.py builds only member 0 on CPU workers
            member['cpu_size'] = list(args.cpu_size)
        if i == 0 and args.cpu_cc is not None:
            cpu_size = tuple(args.cpu_size or size)
            member['cpu_cc_lambda'] = args.cpu_cc
            member['cpu_correction'] = calibration(tail, cpu_size, head, args.weights, prior, args.cpu_cc)
        members.append(member)
    (args.out / 'members.json').write_text(json.dumps(
        {'members': members, 'attribute_names': names, 'tta': args.tta,
         'prior': prior.tolist()}, indent=2) + '\n')
    with zipfile.ZipFile(f'{args.out}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.out.iterdir()):
            archive.write(path, path.name)
    print(f'{args.out}.zip', Path(f'{args.out}.zip').stat().st_size // 2**20, 'MiB')


if __name__ == '__main__':
    main()
