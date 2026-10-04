"""Where do per-image (instance) errors come from? For a LODO run's held-out probe: per attribute FP/FN counts and their
share of all instance errors, plus the top confusions inside colour groups (true colour -> predicted colour).
Usage (workspace root): PYTHONPATH=src python src/error_anatomy.py TAG [FOLD]"""
import sys, collections, numpy as np, prepare as P
from train_dg_2027 import read_rows

names, tr = read_rows(P.TRAIN_CSV)
tag, held = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'Market1501')
src = np.asarray([r[1:] for r in tr if not r[0].startswith(held + '/')], dtype=np.float64)
prior = np.clip(src.mean(0), 1e-4, 1 - 1e-4)
d = np.load(f'runs/lodo/{tag}/{held}/probe.npz'); y = d['labels'] > 0
p = (d['logits'] - P.CC_LAMBDA * (d['content_free'].mean(0) - np.log(prior / (1 - prior)))) > 0
fp, fn = (p & ~y).sum(0), (~p & y).sum(0); tot = fp.sum() + fn.sum()
print(f'{tag} {held}: {len(y)} imgs, pred/img {p.sum(1).mean():.2f} true/img {y.sum(1).mean():.2f}, '
      f'FP/img {fp.sum() / len(y):.2f} FN/img {fn.sum() / len(y):.2f}, error-free imgs {np.mean((p == y).all(1)):.3f}')
print('top attributes by share of instance errors (FP+FN):')
for i in np.argsort(-(fp + fn))[:14]:
    print(f'  {names[i]:30s} share {100 * (fp[i] + fn[i]) / tot:5.1f}%  FP {fp[i]:5d}  FN {fn[i]:5d}  pos {y[:, i].sum():5d}')
for part in ('UpperBody-Color-', 'LowerBody-Color-'):
    idx = [i for i, n in enumerate(names) if n.startswith(part)]
    conf = collections.Counter()
    for r in range(len(y)):
        t = [names[i][len(part):] for i in idx if y[r, i] and not p[r, i]]
        q = [names[i][len(part):] for i in idx if p[r, i] and not y[r, i]]
        for a in t:
            for b in q: conf[(a, b)] += 1
    print(f'{part[:-1]} confusions (true -> predicted):', ', '.join(f'{a}->{b} {c}' for (a, b), c in conf.most_common(8)))
