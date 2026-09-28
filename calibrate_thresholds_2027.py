"""Select a global threshold on one half of validation, report on the other."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_cached_head import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probs', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=605)
    args = parser.parse_args()
    probs, labels = np.load(args.probs), np.load(args.labels)
    with args.val_csv.open(newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        paths = [row[0] for row in reader]
    assert len(probs) == len(labels) == len(paths)
    rng = np.random.default_rng(args.seed)
    calibrate_mask = np.zeros(len(paths), dtype=bool)
    for domain in ('Market1501', 'PA100k', 'PETA'):
        indices = np.flatnonzero([path.startswith(domain + '/') for path in paths])
        calibrate_mask[rng.permutation(indices)[:len(indices) // 2]] = True
    eval_mask = ~calibrate_mask
    thresholds = np.round(np.arange(0.10, 0.901, 0.02), 2)
    candidates = [(float(t), score(labels[calibrate_mask], probs[calibrate_mask], t)['Avg'])
                  for t in thresholds]
    chosen, calibration_avg = max(candidates, key=lambda pair: pair[1])
    result = {
        'selection': 'global threshold maximizing Avg on stratified validation half',
        'seed': args.seed, 'calibration_images': int(calibrate_mask.sum()),
        'evaluation_images': int(eval_mask.sum()),
        'threshold': chosen, 'calibration_avg': calibration_avg,
        'evaluation_at_0.5': score(labels[eval_mask], probs[eval_mask]),
        'evaluation_at_selected': score(labels[eval_mask], probs[eval_mask], chosen),
        'domains': {},
    }
    for domain in ('Market1501', 'PA100k', 'PETA'):
        mask = eval_mask & np.fromiter((path.startswith(domain + '/') for path in paths), bool)
        result['domains'][domain] = {'images': int(mask.sum()),
                                     'at_0.5': score(labels[mask], probs[mask]),
                                     'at_selected': score(labels[mask], probs[mask], chosen)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'threshold': chosen,
                      'Avg_at_0.5': result['evaluation_at_0.5']['Avg'],
                      'Avg_at_selected': result['evaluation_at_selected']['Avg']}, indent=2))


if __name__ == '__main__':
    main()
