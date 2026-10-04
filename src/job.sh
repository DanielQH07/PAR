#!/bin/bash
# The 3 leave-one-domain-out folds of one recipe, sequentially. Usage: bash src/job.sh TAG [train.py args...]
# Watchdog: a DataLoader worker can segfault at spawn and leave the parent blocked on a pipe forever (seen 6+ times,
# kernel log "python[...]: segfault"). A fold is killed and retried once when its log has not changed for STALL_S
# seconds (default 720; an epoch logs every ~2-5 min) or when it runs longer than FOLD_TIMEOUT (default 150m).
cd "$(dirname "$0")/.."  # repository root (expects the ann/ and images/ links, see README.md)
mkdir -p logs
PY=${PY:-python}
TAG=$1; shift
STALL_S=${STALL_S:-720}
FT=${FOLD_TIMEOUT:-150m}
LIMIT_S=$(( ${FT%m} * 60 ))
for d in Market1501 PA100k PETA; do
  out=runs/lodo/$TAG/$d
  log="logs/$TAG.$d.log"
  for attempt in 1 2; do
    rm -rf "$out"
    PYTHONPATH=src $PY src/train.py \
      --weights weights/clip_b16 --train-csv ann/train/gt.csv --val-csv ann/val/gt.csv \
      --image-root images --output-dir "$out" --held-out-domain "$d" "$@" > "$log" 2>&1 &
    pid=$!; start=$(date +%s); reason=""
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      now=$(date +%s); last=$(stat -c %Y "$log")
      if [ $((now - last)) -gt $STALL_S ]; then reason="log stalled $((now - last)) s"; break; fi
      if [ $((now - start)) -gt $LIMIT_S ]; then reason="over $LIMIT_S s"; break; fi
    done
    if [ -n "$reason" ]; then
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; sleep 10
      pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null
      echo "fold $d killed ($reason, attempt $attempt)" >> "logs/$TAG.$d.attempts"
      continue
    fi
    wait $pid; rc=$?
    [ $rc -eq 0 ] && break
    echo "fold $d failed (exit $rc, attempt $attempt)" >> "logs/$TAG.$d.attempts"
  done
  [ -n "$KEEP_PTH" ] || rm -f "$out"/*.pth
done
touch "logs/$TAG.done"
