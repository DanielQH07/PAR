#!/bin/bash
# v8 on the training server: member 0 = uniform soup of 3 seeds of the recipe chosen on LODO (lodo_robust),
# trained with src/train.py on the full train split (epoch selected on val, as v6/v7), packaged in the
# v5 runtime at 256x128, then the same pre-submission checks as v7.
# Usage (workspace root): V8_ARGS="--caption-loss 0.5" CC=0.1 tmux new -d -s v8 'bash src/v8_pipeline.sh'
set -euo pipefail
PY=${PY:-python}
V8_ARGS=${V8_ARGS:-}
CC=${CC:-0.1}
NAME=${NAME:-v8}
# pretrained weights used for training AND packaging (shared.pth holds the frozen embeddings/pre-LN, so it must match).
# With non-OpenAI weights only member 0 is packaged: the legacy/text members were trained on OpenAI CLIP.
WEIGHTS=${WEIGHTS:-weights/clip_b16}
IMG=$(readlink -f images)
LOG=runs/${NAME}_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) start  weights: $WEIGHTS  args: $V8_ARGS  cc: $CC"

runs=()
for s in 605 606 607; do
  out=runs/full_${NAME}_s$s
  for attempt in 1 2 3; do  # watchdog: DataLoader worker segfault at spawn can hang the parent forever
    [ -f $out/metrics.json ] && break
    rm -rf $out
    timeout --kill-after=60 ${SEED_TIMEOUT:-120m} env PYTHONPATH=src $PY src/train.py --weights $WEIGHTS \
      --train-csv ann/train/gt.csv --val-csv ann/val/gt.csv --image-root images --workers 4 --seed $s $V8_ARGS \
      --output-dir $out > $out.log 2>&1 || echo "seed $s attempt $attempt failed ($?)"
  done
  [ -f $out/metrics.json ] || { echo "seed $s failed 3 times"; exit 1; }
  grep '^{' $out.log | tail -1
  runs+=($out)
done
echo "== $(date) trained"

rm -rf runs/full_${NAME}_soup3 && mkdir runs/full_${NAME}_soup3
RUNS="${runs[*]}" OUT=runs/full_${NAME}_soup3 $PY - <<'EOF'
import json, os, torch
runs, out = os.environ['RUNS'].split(), os.environ['OUT']
states = [torch.load(f'{r}/best_ema.pth', map_location='cpu', weights_only=True) for r in runs]
assert all(s.keys() == states[0].keys() for s in states)
soup = {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype)
        if states[0][k].is_floating_point() else states[0][k] for k in states[0]}
torch.save(soup, f'{out}/best_ema.pth')
meta = json.load(open(f'{runs[0]}/metrics.json'))
meta.update(output_dir=out, soup_of=runs,
            source_val_best=[json.load(open(f'{r}/metrics.json'))['source_val_best'] for r in runs])
json.dump(meta, open(f'{out}/metrics.json', 'w'), indent=2)
print('soup of', runs, 'source val best', meta['source_val_best'])
EOF

rm -rf builds/ensemble_${NAME} builds/ensemble_${NAME}.zip
$PY src/package_submission.py --weights $WEIGHTS --val-csv ann/val/gt.csv --train-csv ann/train/gt.csv \
  --tta $( [ "$WEIGHTS" = weights/clip_b16 ] && echo "--cc $CC 0 0.1 --member dg:runs/full_${NAME}_soup3 \
  --member legacy:runs/full_baseline/best_model.pth --member dg:runs/full_text" || echo "--cc $CC --member dg:runs/full_${NAME}_soup3" ) \
  --out builds/ensemble_${NAME}
chmod a+r builds/ensemble_${NAME}.zip
cmp builds/ensemble_${NAME}/run.py submission/task1/run.py && echo "run.py = v5 runtime"
sha256sum builds/ensemble_${NAME}.zip

echo "== in-domain val (3000 imgs) member 0, CPU path at 256x128 (sanity only; v7 = 0.8718)"
$PY src/check_cpu_size.py --build builds/ensemble_${NAME} --val-csv ann/val/gt.csv --image-root images --limit 3000

echo "== CPU check: official image, --cpus 6, non-root, no network (600 val images)"
rm -rf sim/${NAME}in && mkdir -p sim/${NAME}in
$PY src/codabench_ingestion_sim.py --zip x --input-dir sim/${NAME}in --work-dir x --output x \
  --build-from-csv ann/val/gt.csv --limit 600
$PY -c "import json,sys;p='sim/${NAME}in/task.json';t=json.load(open(p));t['num_images']=29248;json.dump(t,open(p,'w'))"
rm -f sim/${NAME}in/images && mkdir -p sim/${NAME}in/images && chmod -R a+rX sim/${NAME}in && mkdir -p sim/out && chmod 777 sim/out
docker run --rm --network none --cpus 6 -m 8g --user 65534:65534 -e HOME=/tmp \
  -v "$PWD/builds/ensemble_${NAME}.zip:/app/sub.zip:ro" -v "$PWD/sim/${NAME}in:/app/input_data:ro" \
  -v "$IMG:/app/input_data/images:ro" -v "$PWD/src/codabench_ingestion_sim.py:/app/ingest.py:ro" \
  -v "$PWD/sim/out:/out" -w /tmp pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  python /app/ingest.py --zip /app/sub.zip --input-dir /app/input_data --work-dir /tmp/work --output /out/${NAME}_cpu.csv
rm -rf builds/ensemble_${NAME} sim/${NAME}in
echo "== $(date) done" && touch runs/${NAME}.done
