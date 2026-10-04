"""Stricter stand-in for the Codabench ingestion program (the real one is private).

Follows the layout documented on the competition's Data/Submission pages:
  input_data/images.csv   one image path per line, no labels
  input_data/images/...   files at those paths
  input_data/task.json    attribute names, number of images
It unzips the submission itself, chdirs into it, `import run`, calls load_model()
if present, then predict_batch (BATCH_SIZE) or predict_image, validates every row
and writes predictions.csv. Any exception is printed with its traceback and exits 1.
Only needs the Python standard library + whatever run.py imports.
"""

import argparse
import csv
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path


def build_input(csv_path, image_root, out, limit):
    """Create input_data/ from a labelled CSV (symlinks, no copying)."""
    with open(csv_path, newline='') as stream:
        rows = list(csv.reader(stream))
    names, rows = rows[0][1:], rows[1:]
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
    if image_root and not (out / 'images').exists():  # host runs; in Docker mount images/ instead
        (out / 'images').symlink_to(Path(image_root).resolve())
    (out / 'images.csv').write_text(''.join(r[0] + '\n' for r in rows))
    (out / 'task.json').write_text(json.dumps({'attribute_names': names, 'num_images': len(rows)}))


def ingest(zip_path, input_dir, work_dir, output):
    sub = (work_dir / 'submission').resolve()
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(sub)
    assert (sub / 'run.py').exists(), 'run.py must be at the archive root'
    input_dir, output = input_dir.resolve(), output.resolve()
    task = json.loads((input_dir / 'task.json').read_text())
    names = task['attribute_names']
    paths = (input_dir / 'images.csv').read_text().split()
    samples = [{'index': i, 'image': p, 'image_path': str((input_dir / 'images' / p).resolve()),
                'attribute_names': names} for i, p in enumerate(paths)]
    sys.path.insert(0, str(sub))
    import os
    os.chdir(sub)
    started = time.monotonic()
    import run  # noqa: E402  (same as the platform: import from the submission dir)
    if hasattr(run, 'load_model'):
        run.load_model()
    loaded = time.monotonic()
    rows = []
    if hasattr(run, 'predict_batch'):
        size = int(getattr(run, 'BATCH_SIZE', 32))
        for offset in range(0, len(samples), size):
            rows.extend(run.predict_batch(samples[offset:offset + size]))
    else:
        rows = [run.predict_image(s) for s in samples]
    finished = time.monotonic()
    assert len(rows) == len(samples), (len(rows), len(samples))
    with open(output, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image', *names])
        for sample, row in zip(samples, rows):
            if isinstance(row, dict):
                row = [row[n] for n in names]
            row = [float(v) for v in list(row)]
            assert len(row) == 40, len(row)
            assert all(0.0 <= v <= 1.0 for v in row), row  # also rejects NaN
            writer.writerow([sample['image'], *row])
    per_image = (finished - loaded) / max(1, len(samples))
    print(f'ok: {len(samples)} images, load {loaded - started:.1f}s, '
          f'predict {finished - loaded:.1f}s ({per_image * 1000:.1f} ms/image, '
          f'~{per_image * 29248 / 60:.0f} min for 29,248 images)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--build-from-csv', type=Path, help='create --input-dir from a labelled CSV first')
    p.add_argument('--image-root', type=Path)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.build_from_csv:
        args.input_dir.mkdir(parents=True, exist_ok=True)
        build_input(args.build_from_csv, args.image_root, args.input_dir, args.limit)
        return
    try:
        ingest(args.zip, args.input_dir, args.work_dir, args.output)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
