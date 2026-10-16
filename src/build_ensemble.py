"""Build the per-attribute weighted submission (the v23/v24 layout): one MetaCLIP member on all 40 attributes plus
TorchScript PE members, each assigned to the colour attributes or to the other attributes.

--base is a single-member build from package_submission.py, as a directory or .zip (members.json, shared.pth,
member_0.pth, clip_vit.py, metadata.yaml). Each --pe is FILE:GROUP with GROUP in {colour, other}; FILE needs export_report.json next to it
(written by export_pe.py). Weights: MetaCLIP 0.5 everywhere, the PE member of a group 0.5 on that group.

Usage (repository root):
  PYTHONPATH=src python src/build_ensemble.py --base builds/ensemble_t2.zip --out builds/ensemble_v24 \
    --pe exports/pe606/pe606.pt:colour --pe exports/pel280/pel280.pt:other \
    --bias Age-Adult=0.005 --bias LowerBody-Length-Short=0.005
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pe', action='append', required=True, help='FILE:GROUP, GROUP in {colour, other}')
    p.add_argument('--bias', action='append', default=[], help='ATTRIBUTE=VALUE added to the final probability')
    p.add_argument('--tta', action='store_true', help='h-flip TTA (off in v23/v24)')
    args = p.parse_args()
    if args.base.suffix == '.zip':  # v8_pipeline.sh keeps only the zip
        tmp = Path(tempfile.mkdtemp())
        shutil.unpack_archive(args.base, tmp)
        args.base = tmp

    config = json.loads((args.base / 'members.json').read_text())
    assert len(config['members']) == 1, 'base must be a single MetaCLIP member build'
    names = config['attribute_names']
    colour = [('Color-' in n) for n in names]
    shutil.rmtree(args.out, ignore_errors=True)
    shutil.copytree(args.base, args.out, ignore=shutil.ignore_patterns('*.zip', '__pycache__'))
    shutil.copy(ROOT / 'submission/task1/run.py', args.out / 'run.py')

    clip = config['members'][0]
    clip['attribute_weight'] = [0.5] * 40
    members = [clip]
    for spec in args.pe:
        file, group = spec.rsplit(':', 1)
        file = Path(file)
        assert group in ('colour', 'other'), spec
        report = json.loads((file.parent / 'export_report.json').read_text())
        shutil.copy(file, args.out / file.name)
        members.append({'file': file.name, 'format': 'torchscript', 'size': report['size'], 'head': 'linear',
                        'source': str(file), 'cc_lambda': 0.1, 'correction': report['correction'],
                        'attribute_weight': [0.5 if c == (group == 'colour') else 0.0 for c in colour]})
    total = [sum(m['attribute_weight'][i] for m in members) for i in range(40)]
    assert all(abs(t - 1) < 1e-9 for t in total), 'each attribute needs exactly one PE member (colour + other)'

    bias = [0.0] * 40
    for item in args.bias:
        name, value = item.split('=')
        bias[names.index(name)] = float(value)
    config['members'] = members
    config['probability_bias'] = bias
    config['tta'] = args.tta
    (args.out / 'members.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.make_archive(str(args.out), 'zip', args.out)
    print(f'{args.out}.zip: {len(members)} members, bias on {sum(b != 0 for b in bias)} attributes')


if __name__ == '__main__':
    main()
