"""Mimic the Codabench ingestion program: import <submission>/run.py, call
load_model() and predict_batch() on sample dicts, save the probabilities.
Needs only numpy + torch + Pillow, so it runs inside the evaluation image."""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    with args.csv.open(newline='') as stream:
        rows = list(csv.reader(stream))
    names, rows = rows[0][1:], rows[1:]
    if args.limit:
        rows = rows[::max(1, len(rows) // args.limit)][:args.limit]  # spread over domains
    sys.path.insert(0, str(args.submission))
    spec = importlib.util.spec_from_file_location('run', args.submission / 'run.py')
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    started = time.monotonic()
    run.load_model()
    samples = [{'index': i, 'image': r[0], 'image_path': str(args.image_root / r[0]),
                'attribute_names': names} for i, r in enumerate(rows)]
    batch = getattr(run, 'BATCH_SIZE', 32)
    probs = []
    for offset in range(0, len(samples), batch):
        out = np.asarray(run.predict_batch(samples[offset:offset + batch]), dtype=np.float64)
        assert out.shape == (len(samples[offset:offset + batch]), 40), out.shape
        assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1).all()
        probs.append(out)
    probs = np.concatenate(probs)
    np.save(args.output, probs)
    np.save(args.output.with_name(args.output.stem + '_labels.npy'),
            np.asarray([r[1:] for r in rows], dtype=np.float32))
    (args.output.with_name(args.output.stem + '_paths.txt')).write_text('\n'.join(r[0] for r in rows))
    print(f'{len(rows)} images in {time.monotonic() - started:.1f}s -> {args.output}')


if __name__ == '__main__':
    main()
