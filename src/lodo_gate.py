"""Compare two ensemble layouts on the 3 source LODO folds (the gate used to pick v23/v24).

Each --expert NAME=DIR points at a LODO run with DIR/<fold>/{probe.npz,metrics.json} (written by train.py with
--held-out-domain). A layout says which experts are averaged on the colour attributes and on the others, e.g.
"colour=t2+pe606 other=t2+pel". Probabilities use the submission's contextual calibration (lambda 0.1).
A candidate passes when Challenge_Avg rises and instance recall and colour-attribute mA do not fall on every fold.

Usage (repository root):
  PYTHONPATH=src python src/lodo_gate.py --expert t2=runs/lodo/T2 --expert pe606=runs/lodo/PE606 \
    --expert pel=runs/lodo/PEL280 --baseline "colour=t2+pe606 other=t2+pe606" \
    --candidate "colour=t2+pe606 other=t2+pel" [--bias Age-Adult=0.005]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_cached_head import challenge_score

FOLDS = ('Market1501', 'PA100k', 'PETA')


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    return rows[0][1:], [r[0] for r in rows[1:]], np.asarray([r[1:] for r in rows[1:]], dtype=float)


def probability(folder, labels, prior, fold, lam=0.1):
    if json.loads((folder / 'metrics.json').read_text()).get('held_out_domain') != fold:
        raise ValueError(f'{folder}: held-out domain mismatch')
    with np.load(folder / 'probe.npz') as probe:
        if not np.array_equal(labels, probe['labels']):
            raise ValueError(f'{folder}: labels/order mismatch')
        offset = lam * (probe['content_free'].mean(0) - np.log(prior / (1 - prior)))
        return 1 / (1 + np.exp(-(probe['logits'] - offset)))


def layout(spec):
    parts = dict(item.split('=') for item in spec.split())
    return parts['colour'].split('+'), parts['other'].split('+')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expert', action='append', required=True, help='NAME=LODO_RUN_DIR')
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--bias', action='append', default=[], help='ATTRIBUTE=VALUE (applied to both layouts)')
    p.add_argument('--train-csv', default='ann/train/gt.csv')
    p.add_argument('--val-csv', default='ann/val/gt.csv')
    args = p.parse_args()

    experts = {k: Path(v) for k, v in (e.split('=') for e in args.expert)}
    names, train_paths, train_y = read_csv(args.train_csv)
    _, val_paths, val_y = read_csv(args.val_csv)
    train_dom = np.asarray([x.split('/')[0] for x in train_paths])
    val_dom = np.asarray([x.split('/')[0] for x in val_paths])
    colour = np.asarray(['Color-' in n for n in names])
    bias = np.zeros(40)
    for item in args.bias:
        name, value = item.split('=')
        bias[names.index(name)] = float(value)

    report, passes = {}, True
    for fold in FOLDS:
        y = val_y[val_dom == fold]
        prior = np.clip(train_y[train_dom != fold].mean(0), 1e-4, 1 - 1e-4)
        probs = {k: probability(d / fold, y, prior, fold) for k, d in experts.items()}
        mix = lambda spec: np.clip(np.where(colour, *(np.mean([probs[k] for k in ks], 0) for ks in layout(spec)))
                                   + bias, 0, 1)
        cols = [i for i in np.flatnonzero(colour) if y[:, i].sum() >= 10]

        def colour_ma(q):
            pred, truth = q[:, cols] >= .5, y[:, cols] == 1
            return float(.5 * ((pred & truth).sum(0) / truth.sum(0) + (~pred & ~truth).sum(0) / (~truth).sum(0)).mean())

        base, cand = mix(args.baseline), mix(args.candidate)
        sb, sc = (challenge_score(y, q, [fold] * len(y)) for q in (base, cand))
        r = {'baseline_H': sb['Challenge_Avg'], 'candidate_H': sc['Challenge_Avg'],
             'dH': sc['Challenge_Avg'] - sb['Challenge_Avg'], 'dRecall': sc['Inst_Rec'] - sb['Inst_Rec'],
             'dColour_mA': colour_ma(cand) - colour_ma(base)}
        passes &= r['dH'] > 0 and r['dRecall'] >= 0 and r['dColour_mA'] >= 0
        report[fold] = r
        print(fold, ' '.join(f'{k} {v:+.5f}' if k.startswith('d') else f'{k} {v:.5f}' for k, v in r.items()))
    print('PASS' if passes else 'FAIL')


if __name__ == '__main__':
    main()
