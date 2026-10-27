"""Uniform weight soup of several train.py runs (same as src/v8_pipeline.sh). Usage: python src/soup.py OUT RUN [RUN ...]"""
import json, sys, torch
from pathlib import Path
out, runs = Path(sys.argv[1]), sys.argv[2:]
out.mkdir(parents=True, exist_ok=True)
states = [torch.load(f'{r}/best_ema.pth', map_location='cpu', weights_only=True) for r in runs]
assert all(s.keys() == states[0].keys() for s in states)
soup = {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype)
        if states[0][k].is_floating_point() else states[0][k] for k in states[0]}
torch.save(soup, out / 'best_ema.pth')
meta = json.load(open(f'{runs[0]}/metrics.json'))
meta.update(output_dir=str(out), soup_of=runs)
json.dump(meta, open(out / 'metrics.json', 'w'), indent=2)
print('soup of', runs)
