"""Weight-average (uniform soup, DiWA-style) several train_dg_2027.py runs that share the
same initialisation, and write the result as a new run directory for probing."""
import argparse
import json
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    states = [torch.load(r / 'best_ema.pth', map_location='cpu', weights_only=True) for r in args.runs]
    keys = set(states[0])
    assert all(set(s) == keys for s in states), 'runs must share the same architecture'
    soup = {k: (sum(s[k].float() for s in states) / len(states)) if states[0][k].is_floating_point()
            else states[0][k] for k in keys}
    args.out.mkdir(parents=True, exist_ok=False)
    torch.save(soup, args.out / 'best_ema.pth')
    meta = json.loads((args.runs[0] / 'metrics.json').read_text())
    meta['soup_of'] = [str(r) for r in args.runs]
    (args.out / 'metrics.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
