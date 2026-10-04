"""mA vs instance metrics per held-out fold (robust2 attribute set) for several LODO runs.
Usage (workspace root): PYTHONPATH=src python src/metric_split.py TAG [TAG ...]"""
import sys, numpy as np, prepare as P
from train_dg_2027 import read_rows

_, tr = read_rows(P.TRAIN_CSV)
for tag in sys.argv[1:]:
    out = []
    for held in ('Market1501', 'PA100k', 'PETA'):
        src = np.asarray([r[1:] for r in tr if not r[0].startswith(held + '/')], dtype=np.float64)
        prior, ks = np.clip(src.mean(0), 1e-4, 1 - 1e-4), src.sum(0) >= P.MIN_SOURCE_POSITIVES
        d = np.load(f'runs/lodo/{tag}/{held}/probe.npz'); y = d['labels']
        z = d['logits'] - P.CC_LAMBDA * (d['content_free'].mean(0) - np.log(prior / (1 - prior)))
        k = np.flatnonzero(ks & (y.sum(0) >= 50))
        m = P.challenge_score(y[:, k], 1 / (1 + np.exp(-z[:, k])), [held] * len(y))
        out.append(f"{held[:6]} H {m['Challenge_Avg']:.4f} mA {m['mA']:.4f} F1 {m['Inst_F1']:.4f} P {m['Inst_Prec']:.4f} R {m['Inst_Rec']:.4f}")
    print(f'{tag:14s} ' + ' | '.join(out))
