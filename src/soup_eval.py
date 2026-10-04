"""S2: weight soup of a short (2-epoch) and a long (6-epoch) run of the same LODO fold, scored with
prepare.evaluate_heldout. Usage (workspace root):
  PYTHONPATH=src python src/soup_eval.py RUN_A RUN_B   (fold dirs, e.g. .../P1metaclip/PETA)
Prints heldout robust for alpha (weight of RUN_A) in 0, 0.3, 0.5, 0.7, 1."""
import json, sys
from pathlib import Path
import numpy as np, torch
import prepare
from train import PARPlus
from train_dg_2027 import read_rows


def main():  # spawn DataLoader workers re-import this module: keep work under the main guard
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((a / 'metrics.json').read_text())
    held, size = meta['held_out_domain'], tuple(meta['size'])
    sa, sb = (torch.load(r / 'best_ema.pth', map_location='cpu', weights_only=True) for r in (a, b))
    assert sa.keys() == sb.keys()
    _, rows = read_rows('ann/train/gt.csv')
    prior = np.asarray([r[1:] for r in rows if not r[0].startswith(held + '/')], dtype=np.float64).mean(0)
    device = torch.device('cuda')
    model = PARPlus(size, meta['head'])
    model.vision.load_hf_state({k: v.float() for k, v in torch.load(Path(meta['weights']) / 'pytorch_model.bin',
                                                                      map_location='cpu', weights_only=True).items()})
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        mix = {k: (alpha * sa[k].float() + (1 - alpha) * sb[k].float()) if sa[k].is_floating_point() else sa[k] for k in sa}
        model.load_state_dict(mix, strict=False)
        r = prepare.evaluate_heldout(model.to(device), size, held, prior, device, workers=2)
        print(f'{held} alpha({a.parent.name})={alpha}: robust {r["cc_robust"]:.4f} cc {r["cc"]["Challenge_Avg"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
