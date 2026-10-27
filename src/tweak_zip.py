"""Variant of an already built submission dir (builds/ensemble_*): same weights files (hard links), edited
members.json, current submission/task1/run.py; zipped and saved under a fixed id.
Usage (repo root): python src/tweak_zip.py BASE_DIR NAME ID [options]
  --bias ATTR=V        add V to that attribute's final probability (repeatable, on top of the base bias)
  --global-bias V      add V to every attribute
  --bias-json F:S      add S * the per-attribute biases stored in JSON file F ({attribute: value})
  --cc S               scale every member's calibration offset by S (S=2 -> lambda 0.2 when shipped with 0.1)
  --member-tta FILE    h-flip TTA for that member only (repeatable)
  --colour-weights F=W,...  per-member weight on the colour attributes; members not listed get 0 there
  --other-weights F=W,...   same for the other attributes"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from save_submission import save

ROOT = Path(__file__).resolve().parents[1]


def weights(spec):
    return {f: float(w) for f, w in (item.split('=') for item in spec.split(','))}


p = argparse.ArgumentParser()
p.add_argument('base', type=Path)
p.add_argument('name')
p.add_argument('id', type=int)
p.add_argument('--bias', action='append', default=[])
p.add_argument('--global-bias', type=float, default=0.0)
p.add_argument('--bias-json')
p.add_argument('--cc', type=float, default=1.0)
p.add_argument('--member-tta', action='append', default=[])
p.add_argument('--colour-weights', type=weights)
p.add_argument('--other-weights', type=weights)
args = p.parse_args()

config = json.loads((args.base / 'members.json').read_text())
names = config['attribute_names']
bias = [b + args.global_bias for b in config.get('probability_bias', [0.0] * 40)]
for item in args.bias:
    name, value = item.split('=')
    bias[names.index(name)] += float(value)
if args.bias_json:
    file, scale = args.bias_json.rsplit(':', 1)
    for name, value in json.loads(Path(file).read_text()).items():
        bias[names.index(name)] += float(scale) * value
config['probability_bias'] = bias
files = [m['file'] for m in config['members']]
assert set(args.member_tta) <= set(files), (args.member_tta, files)
for m in config['members']:
    m['correction'] = [c * args.cc for c in m['correction']]
    m['cc_lambda'] = m.get('cc_lambda', 0.1) * args.cc
    if m['file'] in args.member_tta:
        m['tta'] = True
if args.colour_weights or args.other_weights:
    colour = ['Color-' in n for n in names]
    for m in config['members']:
        m['attribute_weight'] = [(args.colour_weights if c else args.other_weights).get(m['file'], 0.0)
                                 for c in colour]
    for i in range(40):
        t = sum(m['attribute_weight'][i] for m in config['members'])
        assert t > 0, names[i]
        for m in config['members']:
            m['attribute_weight'][i] /= t

out = ROOT / 'builds' / f'ensemble_{args.name}'
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
for f in args.base.iterdir():
    if f.name not in ('members.json', 'run.py') and f.is_file():
        os.link(f.resolve(), out / f.name)
shutil.copy(ROOT / 'submission/task1/run.py', out / 'run.py')
(out / 'members.json').write_text(json.dumps(config, indent=2) + '\n')
shutil.make_archive(str(out), 'zip', out)
os.environ['UPAR_SUBMIT_ID'] = str(args.id)
note = ' '.join(sys.argv[4:]) or 'copy'
print(f'{out}.zip: bias on {sum(abs(b) > 1e-12 for b in bias)} attributes, cc x{args.cc}, member tta {args.member_tta}')
save(f'{out}.zip', args.name, f'variant of {args.base.name}: {note}')
