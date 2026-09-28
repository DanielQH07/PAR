"""Compute pooled and domain-specific metrics from a saved best validation pass."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_cached_head import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    args = parser.parse_args()
    labels = np.load(args.run_dir / 'best_val_labels.npy')
    probs = np.load(args.run_dir / 'best_val_probs.npy')
    with args.val_csv.open(newline='') as stream:
        rows = list(csv.reader(stream))
    paths = [row[0] for row in rows[1:]]
    assert len(paths) == len(labels) == len(probs)
    report = {'pooled': score(labels, probs), 'domains': {}}
    for domain in ('Market1501', 'PA100k', 'PETA'):
        mask = np.fromiter((path.startswith(domain + '/') for path in paths), bool)
        report['domains'][domain] = {'images': int(mask.sum()),
                                     **score(labels[mask], probs[mask])}
    (args.run_dir / 'best_val_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
