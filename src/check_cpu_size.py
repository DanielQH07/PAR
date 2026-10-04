"""Check that member 0 of a packaged submission keeps its accuracy at the reduced
CPU input size: score it on validation images at `size` and at `cpu_size`."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cached_head import challenge_score  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True, help='unzipped submission dir')
    p.add_argument('--val-csv', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--limit', type=int, default=3000)
    args = p.parse_args()
    sys.path.insert(0, str(args.build))
    import run  # the submission's own preprocessing (_canvases) and model code
    from clip_vit import CLIP_MEAN, CLIP_STD, PARModel
    config = json.loads((args.build / 'members.json').read_text())
    member = config['members'][0]
    shared = {k: v.float() for k, v in torch.load(args.build / 'shared.pth', weights_only=True).items()}
    tail = {k: v.float() for k, v in torch.load(args.build / member['file'], weights_only=True).items()}
    rows = list(csv.reader(args.val_csv.open()))[1:]
    rows = rows[::max(1, len(rows) // args.limit)][:args.limit]
    labels = np.array([[float(v) for v in r[1:]] for r in rows])
    domains = [r[0].split('/')[0] for r in rows]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_size = member.get('cpu_size', member['size'])
    combos = [(member['size'], 'correction'), (cpu_size, 'correction')]  # size effect at the same lambda
    if 'cpu_correction' in member:
        combos += [(member['size'], 'cpu_correction'), (cpu_size, 'cpu_correction')]  # lambda effect
    for size, key in combos:
        model = PARModel(tuple(size), member['head'])
        model.vision.load_hf_state(shared)
        model.load_state_dict(tail, strict=False)
        model.eval().to(device)
        correction = torch.tensor(member.get(key, member['correction']), device=device)
        probs = []
        with torch.inference_mode():
            for i in range(0, len(rows), 64):
                x = np.stack([run._canvases(str(args.image_root / r[0]), [tuple(size)])[0] for r in rows[i:i + 64]])
                x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255
                x = (x - torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
                probs.append((model(x).float() - correction).sigmoid().cpu().numpy())
        p_all = np.concatenate(probs)
        print(f"member 0 at {tuple(size)} ({key}): val({len(rows)}) Challenge_Avg "
              f"{challenge_score(labels, p_all, domains)['Challenge_Avg']:.4f} / all40 "
              f"{challenge_score(labels, p_all, domains, exclude_invalid=False)['Challenge_Avg']:.4f}", flush=True)


if __name__ == '__main__':
    main()
