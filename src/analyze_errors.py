"""Per-attribute error analysis of the submitted ensemble configuration on LODO probes.
Usage: python src/analyze_errors.py <dir with probe/<model>_<fold>/probe.npz> [lambda]"""
import csv
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from evaluate_cached_head import challenge_score  # noqa: E402

S = sys.argv[1]
LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
FOLDS = ['Market1501', 'PA100k', 'PETA']
MEMBERS = {'base': False, 'linear': True, 'text': True}  # name -> calibrated
names = next(csv.reader(open('data/annotations/task1/val/gt.csv')))[1:]
tr = pd.read_csv('data/annotations/task1/train/gt.csv')
dom = tr.iloc[:, 0].str.split('/').str[0].values
Y = tr.iloc[:, 1:41].values
sig = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(p / (1 - p))

rows = []
for f in FOLDS:
    prior = np.clip(Y[dom != f].mean(0), 1e-4, 1 - 1e-4)
    probs = []
    for m, cal in MEMBERS.items():
        d = np.load(f'{S}/probe/{m}_{f}/probe.npz')
        z = (d['logits'] + d['flipped']) / 2
        if cal:
            z = z - LAM * (d['content_free'].mean(0) - lg(prior))
        probs.append(sig(z))
        y = d['labels']
    p = np.mean(probs, 0)
    pred = p > 0.5
    res = challenge_score(y, p, [f] * len(y))['domains'][f]
    fp_total = ((pred == 1) & (y == 0)).sum()
    fn_total = ((pred == 0) & (y == 1)).sum()
    print(f"{f:10s} Challenge {res['Challenge_Avg']:.4f} mA {res['mA']:.4f} F1 {res['Inst_F1']:.4f} "
          f"P {res['Inst_Prec']:.3f} R {res['Inst_Rec']:.3f} | FP {fp_total} FN {fn_total} "
          f"| pred pos/img {pred.sum(1).mean():.2f} true pos/img {y.sum(1).mean():.2f}")
    for i, n in enumerate(names):
        t, q = y[:, i], pred[:, i]
        if t.sum() == 0:
            continue  # invalid in this domain
        tp, fp, fn, tn = ((q == 1) & (t == 1)).sum(), ((q == 1) & (t == 0)).sum(), ((q == 0) & (t == 1)).sum(), ((q == 0) & (t == 0)).sum()
        rows.append({'fold': f, 'attr': n, 'train_prior': prior[i], 'true_rate': t.mean(), 'pred_rate': q.mean(),
                     'recall': tp / max(tp + fn, 1), 'tnr': tn / max(tn + fp, 1),
                     'ma': (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
                     'fp_share': fp / max(fp_total, 1), 'fn_share': fn / max(fn_total, 1)})
df = pd.DataFrame(rows)
df.to_csv(f'{S}/attr_errors.csv', index=False)
pd.set_option('display.width', 200)
agg = df.groupby('attr').agg(ma=('ma', 'mean'), fp_share=('fp_share', 'mean'), fn_share=('fn_share', 'mean'),
                             true=('true_rate', 'mean'), pred=('pred_rate', 'mean'), prior=('train_prior', 'mean'))
print('\n== worst mA attributes (mean over folds)')
print(agg.sort_values('ma').head(12).round(3).to_string())
print('\n== biggest share of false positives (drives instance precision)')
print(agg.sort_values('fp_share', ascending=False).head(12).round(3).to_string())
print('\n== per fold: attributes predicted far more often than they occur (pred_rate / true_rate > 1.8, true_rate > 1%)')
over = df[(df.pred_rate > 1.8 * df.true_rate) & (df.true_rate > 0.01)]
print(over[['fold', 'attr', 'train_prior', 'true_rate', 'pred_rate', 'fp_share']].round(3).sort_values(['fold', 'fp_share'], ascending=[True, False]).to_string(index=False))
