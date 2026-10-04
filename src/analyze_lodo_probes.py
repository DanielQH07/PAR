"""Score LODO probe logits (from eval_heldout_variants.py) offline: TTA, contextual
calibration with lambda chosen leave-one-fold-out, and fixed ensemble combinations.
Usage: python src/analyze_lodo_probes.py <dir containing probe/<model>_<fold>/probe.npz>
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src'); from evaluate_cached_head import challenge_score
S = sys.argv[1]; folds = ['Market1501', 'PA100k', 'PETA']
tr = pd.read_csv('data/annotations/task1/train/gt.csv'); dom = tr.iloc[:, 0].str.split('/').str[0]; Y = tr.iloc[:, 1:41].values
prior = {f: np.clip(Y[(dom != f).values].mean(0), 1e-4, 1 - 1e-4) for f in folds}
MODELS = ['base', 'base606', 'linear', 'text', 'linu4', 'lintag', 'linanc']
LEGACY = {'base', 'base606'}
P = {(m, f): np.load(f'{S}/probe/{m}_{f}/probe.npz') for m in MODELS for f in folds}
sig = lambda z: 1 / (1 + np.exp(-z)); lg = lambda p: np.log(p / (1 - p))
def score(pr):
    return {f: challenge_score(P[('base', f)]['labels'], pr[f], [f] * len(pr[f]))['Challenge_Avg'] for f in folds}
def variant(m, f, lam):
    d = P[(m, f)]; z = (d['logits'] + d['flipped']) / 2
    if lam and m not in LEGACY: z = z - lam * (d['content_free'].mean(0) - lg(prior[f]))
    return sig(z)
grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
def lofo(fn):
    table = {g: score({f: fn(f, g) for f in folds}) for g in grid}
    pick = {f: max(grid, key=lambda g: np.mean([table[g][h] for h in folds if h != f])) for f in folds}
    return {f: table[pick[f]][f] for f in folds}, pick
fmt = lambda r: f"{np.mean(list(r.values())):.4f} {[round(v, 3) for v in r.values()]}"
print('single models (tta):')
for m in MODELS:
    raw = score({f: sig(P[(m, f)]['logits']) for f in folds})
    r, pick = lofo(lambda f, l: variant(m, f, l))
    print(f"  {m:8s} raw {fmt(raw)} | tta+CC(LOFO) {fmt(r)} lam={sorted(set(pick.values()))}")
combos = {'v4: base+linear+text': ['base', 'linear', 'text'],
          'base+base606 (seeds only)': ['base', 'base606'],
          'base+linear': ['base', 'linear'],
          '+base606': ['base', 'base606', 'linear', 'text'],
          '+linu4': ['base', 'linear', 'text', 'linu4'],
          '+lintag': ['base', 'linear', 'text', 'lintag'],
          'all 6': MODELS,
          'base+base606+linu4+text': ['base', 'base606', 'linu4', 'text']}
print('ensembles (tta, CC on DG members, lam LOFO):')
for name, ms in combos.items():
    r, pick = lofo(lambda f, l: np.mean([variant(m, f, l) for m in ms], 0))
    print(f"  {name:28s} {fmt(r)} lam={sorted(set(pick.values()))}")
