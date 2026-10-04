"""Fixed held-out evaluation; keep it unchanged so that runs stay comparable.

Scores one leave-one-domain-out (LODO) fold the way the shipped submission runs on the
CPU-only Codabench server: a single model, no flip TTA, contextual calibration
(Zhao et al. 2021) with lambda = CC_LAMBDA, threshold 0.5, Codabench metric
(harmonic mean of mA and instance F1; attributes without positives in the domain excluded).

train.py calls evaluate_heldout(...) at the end of a fold and prints the summary lines.
Runs on the training server from the workspace root (ann/, images/, src/ on PYTHONPATH).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_heldout_variants import content_free  # src/: fixed content-free inputs
from evaluate_cached_head import challenge_score  # src/: Codabench metric
from train_dg_2027 import Images, read_rows  # src/: fixed eval preprocessing

DOMAINS = ('Market1501', 'PA100k', 'PETA')
VAL_CSV = 'ann/val/gt.csv'
TRAIN_CSV = 'ann/train/gt.csv'
MIN_SOURCE_POSITIVES = 20  # robust score: skip attributes (near-)absent from the fold's source training data
IMAGE_ROOT = 'images'
CC_LAMBDA = 0.1


@torch.inference_mode()
def evaluate_heldout(model, size, held, prior, device, workers=4, save=None):
    """model: nn.Module mapping CLIP-normalised images (B, 3, H, W) to 40 logits.
    prior: training-fold positive rate per attribute (used by the calibration)."""
    assert held in DOMAINS
    _, rows = read_rows(VAL_CSV)
    rows = [r for r in rows if r[0].startswith(held + '/')]
    loader = DataLoader(Images(rows, IMAGE_ROOT, tuple(size)), batch_size=256, num_workers=workers,
                        multiprocessing_context='spawn' if workers else None)
    model.eval()
    logits, labels = [], []
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for x, y, _ in loader:
            logits.append(model(x.to(device)).float().cpu())
            labels.append(y)
        cf_all = model(content_free(tuple(size)).to(device)).float().cpu().numpy()
    cf = cf_all.mean(0)
    logits, labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    if save is not None:  # probe for Track B (same layout as src/eval_heldout_variants.py, no flip)
        np.savez(save, logits=logits, labels=labels, content_free=cf_all, held=held)
    prior = np.clip(np.asarray(prior, dtype=np.float64), 1e-4, 1 - 1e-4)
    correction = CC_LAMBDA * (cf - np.log(prior / (1 - prior)))
    domains = [held] * len(labels)
    _, train_rows = read_rows(TRAIN_CSV)
    source_pos = np.asarray([r[1:] for r in train_rows if not r[0].startswith(held + '/')], dtype=np.float64).sum(0)
    keep = np.flatnonzero(source_pos >= MIN_SOURCE_POSITIVES)
    out = {}
    for name, z in (('raw', logits), ('cc', logits - correction)):
        m = challenge_score(labels, 1 / (1 + np.exp(-z)), domains)
        out[name] = {k: m[k] for k in ('Challenge_Avg', 'mA', 'Inst_F1', 'Inst_Prec', 'Inst_Rec')}
    z = (logits - correction)[:, keep]
    out['cc_robust'] = challenge_score(labels[:, keep], 1 / (1 + np.exp(-z)), domains)['Challenge_Avg']
    out['robust_dropped'] = int(40 - len(keep))
    out['pred_pos_per_image'] = float((logits - correction > 0).sum(1).mean())
    out['true_pos_per_image'] = float(labels.sum(1).mean())
    return out


def print_summary(held, result, seconds):
    cc, raw = result['cc'], result['raw']
    print('---')
    print(f"fold:             {held}")
    print(f"heldout_cc:       {cc['Challenge_Avg']:.6f}")
    print(f"heldout_raw:      {raw['Challenge_Avg']:.6f}")
    if 'cc_robust' in result:
        print(f"heldout_robust:   {result['cc_robust']:.6f}   (attributes with <{MIN_SOURCE_POSITIVES} source positives dropped: {result['robust_dropped']})")
    print(f"mA_cc:            {cc['mA']:.6f}")
    print(f"inst_f1_cc:       {cc['Inst_F1']:.6f}")
    print(f"inst_prec_cc:     {cc['Inst_Prec']:.6f}")
    print(f"inst_rec_cc:      {cc['Inst_Rec']:.6f}")
    print(f"pred_pos_img:     {result['pred_pos_per_image']:.3f}")
    print(f"true_pos_img:     {result['true_pos_per_image']:.3f}")
    print(f"fold_seconds:     {seconds:.0f}")
