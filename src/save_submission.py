"""Copy a built submission zip to the upload folder as {id}_{name}.zip (id = next free 3-digit number) and append
a row to index.tsv there (id, file, sha256, date, git commit, source zip, note), so every upload can be traced back.

Usage (repository root):
  python src/save_submission.py builds/ensemble_v24.zip --name v24 [--note "T2 + PE-B + PE-L"]
Folder: $UPAR_SUBMIT_DIR, default /mnt/hdd/shared/quanlm46/UPAR/submission; UPAR_SUBMIT_ID=N forces the id.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT_DIR = Path(os.environ.get('UPAR_SUBMIT_DIR', '/mnt/hdd/shared/quanlm46/UPAR/submission'))


def save(zip_path, name, note=''):
    zip_path = Path(zip_path)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    ids = [int(m.group(1)) for f in SUBMIT_DIR.glob('*.zip') if (m := re.match(r'(\d+)_', f.name))]
    wanted = int(os.environ.get('UPAR_SUBMIT_ID', 0))  # fixed id (planned order) instead of the next free one
    assert not wanted or wanted not in ids, f'id {wanted} already used in {SUBMIT_DIR}'
    dest = SUBMIT_DIR / f'{wanted or max(ids, default=0) + 1:03d}_{re.sub(r"[^A-Za-z0-9.-]+", "-", name)}.zip'
    shutil.copyfile(zip_path, dest)
    dest.chmod(0o664)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    index = SUBMIT_DIR / 'index.tsv'
    if not index.exists():
        index.write_text('id\tfile\tsha256\tdate\tcommit\tsource\tnote\n')
    with index.open('a') as f:
        f.write(f'{dest.name[:3]}\t{dest.name}\t{sha}\t{time.strftime("%Y-%m-%d %H:%M")}\t'
                f'{commit}{"+dirty" if dirty else ""}\t{zip_path.resolve()}\t{note}\n')
    print(f'saved {dest} ({dest.stat().st_size / 2**20:.1f} MB, sha256 {sha[:12]})')
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('zip', type=Path)
    p.add_argument('--name', help='short label (default: zip stem without "ensemble_")')
    p.add_argument('--note', default='')
    args = p.parse_args()
    save(args.zip, args.name or args.zip.stem.removeprefix('ensemble_'), args.note)


if __name__ == '__main__':
    main()
