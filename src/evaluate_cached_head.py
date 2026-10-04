"""Evaluate a frozen C2T backbone and linear head on the 2027 validation split."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from utils.metrics.pedestrian_metrics import get_pedestrian_metrics


def score(labels, probabilities, threshold=0.5):
    result = get_pedestrian_metrics(labels, probabilities, threshold=threshold)
    return {
        'mA': float(result.ma),
        'Label_F1': float(np.mean(result.label_f1)),
        'Inst_Acc': float(result.instance_acc),
        'Inst_Prec': float(result.instance_prec),
        'Inst_Rec': float(result.instance_recall),
        'Inst_F1': float(result.instance_f1),
        'Avg': float((result.ma + result.instance_f1) / 2),
    }


def challenge_score(labels, probabilities, domains, threshold=0.5, exclude_invalid=True):
    """Codabench 2027 ranking: per-domain harmonic mean of mA and Inst_F1, averaged
    over domains. By default attributes without positives in a domain are treated as invalid
    (unannotated in that source dataset) and excluded for that domain; the organisers' exact
    rule is unknown, so pass exclude_invalid=False to score all 40 attributes."""
    domains = np.asarray(domains)
    per_domain = {}
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        y, p = labels[mask], probabilities[mask]
        valid = np.flatnonzero(y.sum(axis=0) > 0) if exclude_invalid else np.arange(y.shape[1])
        result = get_pedestrian_metrics(y, p, threshold=threshold, index=valid)
        ma, f1 = float(result.ma), float(result.instance_f1)
        per_domain[domain] = {'images': int(mask.sum()), 'mA': ma, 'Inst_F1': f1,
                              'Inst_Prec': float(result.instance_prec),
                              'Inst_Rec': float(result.instance_recall),
                              'Challenge_Avg': 2 * ma * f1 / max(ma + f1, 1e-12)}
    keys = ('mA', 'Inst_F1', 'Inst_Prec', 'Inst_Rec', 'Challenge_Avg')
    summary = {key: float(np.mean([d[key] for d in per_domain.values()])) for key in keys}
    return {**summary, 'domains': per_domain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--head-checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--head-type', choices=('linear', 'layernorm_linear'), default='linear')
    parser.add_argument('--feature-slice', choices=('all', 'first_half', 'second_half'),
                        default='all')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--save-probs', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    features = np.load(args.cache_dir / 'val_features.npy', mmap_mode='r')
    labels = np.load(args.cache_dir / 'val_labels.npy', mmap_mode='r')
    if args.feature_slice != 'all':
        assert features.shape[1] % 2 == 0
        midpoint = features.shape[1] // 2
        features = features[:, :midpoint] if args.feature_slice == 'first_half' else features[:, midpoint:]
    with args.val_csv.open(newline='') as stream:
        reader = csv.reader(stream)
        names = next(reader)[1:]
        image_paths = [row[0] for row in reader]
    assert len(features) == len(labels) == len(image_paths)
    assert len(names) == labels.shape[1] == 40
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = (torch.nn.Linear(features.shape[1], labels.shape[1])
            if args.head_type == 'linear' else
            torch.nn.Sequential(torch.nn.LayerNorm(features.shape[1]),
                                torch.nn.Linear(features.shape[1], labels.shape[1]))).to(device)
    source = args.head_checkpoint or args.cache_dir / 'classifier_init.pth'
    state = torch.load(source, map_location='cpu')
    head.load_state_dict(state.get('model', state.get('classifier', state)))
    head.eval()
    start = time.monotonic()
    probs = np.empty(labels.shape, dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, len(features), args.batch_size):
            end = min(offset + args.batch_size, len(features))
            x = torch.from_numpy(np.asarray(features[offset:end]).copy()).to(device)
            probs[offset:end] = head(x).sigmoid().cpu().numpy()
    elapsed = time.monotonic() - start
    truth = np.asarray(labels, dtype=np.float32)
    assert np.isin(truth, (0, 1)).all()
    if args.save_probs:
        np.save(args.output_dir / 'val_probs.npy', probs)
        np.save(args.output_dir / 'val_labels.npy', truth)
    metrics = score(truth, probs)
    domain_metrics = {}
    for domain in ('Market1501', 'PA100k', 'PETA'):
        mask = np.fromiter((path.startswith(domain + '/') for path in image_paths), bool)
        domain_metrics[domain] = {'images': int(mask.sum()), **score(truth[mask], probs[mask])}
    predictions = probs >= 0.5
    with (args.output_dir / 'per_attribute.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('attribute', 'positive_frequency', 'accuracy', 'balanced_accuracy',
                         'precision', 'recall', 'F1', 'threshold', 'ECE'))
        for i, name in enumerate(names):
            y, p, pred = truth[:, i], probs[:, i], predictions[:, i]
            tp = np.sum((y == 1) & pred)
            fp = np.sum((y == 0) & pred)
            fn = np.sum((y == 1) & ~pred)
            tn = np.sum((y == 0) & ~pred)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            negative_recall = tn / max(tn + fp, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            bins = np.minimum((p * 10).astype(int), 9)
            ece = sum(np.mean(bins == b) * abs(float(y[bins == b].mean()) -
                      float(p[bins == b].mean())) for b in range(10) if np.any(bins == b))
            writer.writerow((name, float(y.mean()), float((tp + tn) / len(y)),
                             float((recall + negative_recall) / 2), float(precision),
                             float(recall), float(f1), 0.5, float(ece)))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = {
        'experiment_id': args.experiment_id,
        'git_commit': commit,
        'checkpoint_path': str(source.resolve()),
        'head_type': args.head_type,
        'feature_slice': args.feature_slice,
        'cache_dir': str(args.cache_dir.resolve()),
        'val_csv': str(args.val_csv.resolve()),
        'validation_images': len(truth),
        'evaluation_seconds': elapsed,
        'metrics': metrics,
        'domains': domain_metrics,
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'experiment_id': args.experiment_id, **metrics,
                      'evaluation_seconds': elapsed}, indent=2))


if __name__ == '__main__':
    main()
