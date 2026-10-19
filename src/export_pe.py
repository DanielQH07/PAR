"""Export a fine-tuned PE-Core member (arch pe or pel) to TorchScript for the submission and compute its
contextual-calibration correction. Verifies eager vs traced outputs at batch 1/7/64.
Usage (repository root): UPAR_EXPORT_RUN=runs/full_pel UPAR_EXPORT_OUT=exports/pel UPAR_EXPORT_FILE=pel.pt \\
  PYTHONPATH=src python src/export_pe.py   (optional UPAR_EXPORT_SIZE=H,W re-sizes the positional grid;
  UPAR_EXPORT_CKPT=final.pth exports the WiSE-FT weights saved by train.py --save-alpha instead of best_ema.pth;
  any PARPlus arch works, e.g. a SigLIP 2 run)"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_heldout_variants import content_free
from train import PARPlus
from train_dg_2027 import Images

RUN = Path(os.environ["UPAR_EXPORT_RUN"])
OUT = Path(os.environ["UPAR_EXPORT_OUT"])
ARTIFACT = os.environ.get("UPAR_EXPORT_FILE", "pe.pt")


def main():
    meta = json.loads((RUN / "metrics.json").read_text())
    original_size = tuple(meta["size"])
    size = tuple(map(int, os.environ.get("UPAR_EXPORT_SIZE", ",".join(map(str, original_size))).split(",")))
    arch = meta.get("arch", "pe")
    patch = 14 if arch in ("pel", "mc2l", "l14", "h14", "tipsv2", "aimv2l") else 16
    if len(size) != 2 or any(n <= 0 or n % patch for n in size):
        raise ValueError(f"invalid export size: {size}")
    state = torch.load(RUN / os.environ.get("UPAR_EXPORT_CKPT", "best_ema.pth"), map_location="cpu", weights_only=True)
    if size != original_size:
        old_grid = (original_size[0] // patch, original_size[1] // patch)
        new_grid = (size[0] // patch, size[1] // patch)
        position = state["vision.m.pos_embed"]
        if position.shape[1] != 1 + old_grid[0] * old_grid[1]:
            raise ValueError("unexpected PE positional grid")
        patches = position[:, 1:].transpose(1, 2).reshape(1, position.shape[-1], *old_grid)
        patches = F.interpolate(patches, size=new_grid, mode="bicubic",
                                align_corners=False).flatten(2).transpose(1, 2)
        state["vision.m.pos_embed"] = torch.cat((position[:, :1], patches), dim=1)
    model = PARPlus(size, meta["head"], arch=arch)
    if "pseudo_text" in state:
        model.set_pseudo_word(state["pseudo_text"])
    model.load_state_dict(state, strict=True)
    model = model.eval().cuda()
    OUT.mkdir(parents=True, exist_ok=True)

    with open("ann/val/gt.csv") as stream:
        rows = list(csv.reader(stream))[1:65]
    real = next(iter(DataLoader(Images(rows, "images", size), batch_size=64)))[0].cuda()
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(model, real[:1], check_trace=False).eval())
        checks = {}
        for batch in (1, 7, 64):
            eager = model(real[:batch]).float()
            exported = traced(real[:batch]).float()
            diff = (eager - exported).abs().max().item()
            assert eager.shape == exported.shape == (batch, 40)
            assert torch.isfinite(exported).all() and diff < {"hrnet": 5e-3, "mobileclip2": 3e-2}.get(arch, 1e-4), (batch, diff)  # conv: cuDNN algo differences (mobileclip2 1e-2 on GPU, 1e-4 on CPU)
            checks[str(batch)] = diff
        traced.save(str(OUT / ARTIFACT))

        with open("ann/train/gt.csv") as stream:
            labels = np.asarray([r[1:] for r in list(csv.reader(stream))[1:]], dtype=float)
        prior = np.clip(labels.mean(0), 1e-4, 1 - 1e-4)
        cf = model(content_free(size).cuda()).float().mean(0).cpu().numpy()
        correction = (.1 * (cf - np.log(prior / (1 - prior)))).tolist()

    report = {"batch_max_abs_diff": checks, "correction": correction,
              "artifact": str(OUT / ARTIFACT), "size": list(size)}
    (OUT / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
