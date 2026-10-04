"""Per-block relative weight drift ||W - W0|| / ||W0|| of fine-tuned LODO checkpoints vs the pretrained backbone.
Usage (workspace root): PYTHONPATH=src python src/layer_drift.py WEIGHTS_DIR RUN_DIR [RUN_DIR ...]"""
import sys, collections, torch
from clip_vit import PARModel

base = PARModel((256, 128)).vision
base.load_hf_state({k: v.float() for k, v in torch.load(f'{sys.argv[1]}/pytorch_model.bin', map_location='cpu', weights_only=True).items()})
w0 = {k: v for k, v in base.state_dict().items()}
for run in sys.argv[2:]:
    sd = torch.load(f'{run}/best_ema.pth', map_location='cpu', weights_only=True)
    drift = collections.defaultdict(lambda: [0.0, 0.0])
    for k, v in sd.items():
        if not k.startswith('vision.') or k[7:] not in w0 or v.ndim < 2:
            continue
        name = k[7:]; g = name.split('.')[1] if name.startswith('layers.') else name.split('.')[0]
        d = drift[g if name.startswith('layers.') else 'other']
        d[0] += float((v.float() - w0[name]).pow(2).sum()); d[1] += float(w0[name].pow(2).sum())
    blocks = [f'{(drift[str(i)][0] / drift[str(i)][1]) ** 0.5 * 100:.2f}' for i in range(12)]
    print(f'{run:45s} drift %% per block 1..12: ' + ' '.join(blocks))
