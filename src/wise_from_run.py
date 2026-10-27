"""WiSE-FT (and optional soup) of PARPlus runs after training: (1 - a) * pre-trained + a * mean(fine-tuned) on the
vision weights (the linear head has no pre-trained value). Writes OUT/best_ema.pth + metrics.json for export_pe.py.
Usage: PYTHONPATH=src python src/wise_from_run.py OUT ALPHA PRETRAINED_WEIGHTS RUN [RUN ...]"""
import json, sys, torch
from pathlib import Path
from train import PARPlus
out, alpha, weights, runs = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
meta = json.load(open(f'{runs[0]}/metrics.json'))
model = PARPlus(tuple(meta['size']), meta['head'], arch=meta.get('arch', 'b16'))
wf = weights if weights.is_file() else weights / 'pytorch_model.bin'
model.vision.load_hf_state(torch.load(wf, map_location='cpu', weights_only=True))
pre = model.state_dict()
states = [torch.load(f'{r}/best_ema.pth', map_location='cpu', weights_only=True) for r in runs]
mixed = {}
for k, v in states[0].items():
    if not v.is_floating_point():
        mixed[k] = v; continue
    ft = sum(s[k].float() for s in states) / len(states)
    mixed[k] = ((1 - alpha) * pre[k].float() + alpha * ft if k.startswith('vision.') and k in pre else ft).to(v.dtype)
out.mkdir(parents=True, exist_ok=True)
torch.save(mixed, out / 'best_ema.pth')
meta.update(output_dir=str(out), soup_of=runs, wise_alpha=alpha)
json.dump(meta, open(out / 'metrics.json', 'w'), indent=2)
print('wise', alpha, 'of', runs, '->', out)
