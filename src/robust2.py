# Re-score saved probes: robust (source positives >= 20) and robust2 (also held-out positives >= 50).
import os, numpy as np, prepare as P
from train_dg_2027 import read_rows
_, tr = read_rows(P.TRAIN_CSV)
tags = ["F1samples180k", "T1exact180k", "T2exact180k_val", "W1wb08g25"]

ctx = {}
for held in ('Market1501', 'PA100k', 'PETA'):
    src = np.asarray([r[1:] for r in tr if not r[0].startswith(held + '/')], dtype=np.float64)
    ctx[held] = (np.clip(src.mean(0), 1e-4, 1 - 1e-4), src.sum(0) >= P.MIN_SOURCE_POSITIVES)
rows = []
for t in tags:
    r1, r2 = [], []
    for held in ctx:
        f = f'runs/lodo/{t}/{held}/probe.npz'
        if not os.path.exists(f): break
        d = np.load(f); y = d['labels']; prior, ks = ctx[held]
        z = d['logits'] - P.CC_LAMBDA * (d['content_free'].mean(0) - np.log(prior / (1 - prior)))
        for keep, out in ((ks, r1), (ks & (y.sum(0) >= 50), r2)):
            k = np.flatnonzero(keep)
            out.append(P.challenge_score(y[:, k], 1 / (1 + np.exp(-z[:, k])), [held] * len(y))['Challenge_Avg'])
    if len(r1) == 3:
        rows.append((np.mean(r2), t, r1, r2))
for m2, t, r1, r2 in sorted(rows, reverse=True):
    print(f'{t:16s} robust {np.mean(r1):.4f}  robust2 {m2:.4f}  [' + ' '.join(f'{x:.4f}' for x in r2) + ']')
