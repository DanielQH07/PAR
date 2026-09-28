# UPAR 2027 Track 1 experiment record

Data: official `speckean/UPAR-Challenge-2027` task 1 CSVs, 97,669 train and 33,407 validation images. The CSV rows match the local copies exactly. All labels are binary and all referenced images are present locally. Metric `Avg = (mA + Inst_F1) / 2`, using threshold 0.5 unless stated otherwise. The 2027 hidden test has no labels here.

## C²T checkpoint reference

The published C²T checkpoint yields validation mA 0.85991, Inst_F1 0.89944 and Avg **0.87968**. Retaining the frozen C²T backbone and training its published linear classifier for 10 epochs yields Avg 0.88053 ± 0.00015 across three seeds. A new LayerNorm→Linear head with weighted BCE on the same frozen features yields Avg **0.89049 ± 0.00076** across three seeds. The latter changes both head and loss, so it is a head experiment, not a C²T architecture ablation. This checkpoint is a historical warm start; no independent 2027 end-to-end C²T training has been completed.

## Frozen backbone bake-off

All models use an aspect-preserving resize and centered pad to 224×224; checkpoint-specific mean and standard deviation; frozen pretrained vision weights; LayerNorm→Linear(40); weighted BCE; 50 head epochs; seed 605 for LODO. Pooled scores are mean ± sample standard deviation across three seeds where available. LODO chooses the checkpoint on source-domain validation and evaluates once on the held-out domain.

| Backbone | Pooled Avg | →Market | →PA100k | →PETA | Mean LODO | Worst LODO |
|---|---:|---:|---:|---:|---:|---:|
| CLIP ViT-B/16, preprojection | 0.79319 ± 0.00058 | 0.7352 | 0.7340 | 0.6979 | 0.7224 | 0.6979 |
| SigLIP2 ViT-B/16 | 0.78274 ± 0.00023 | 0.7233 | 0.7409 | 0.6984 | 0.7209 | 0.6984 |
| Swin-B | 0.78108 (1 seed) | 0.7229 | 0.7296 | 0.6700 | 0.7075 | 0.6700 |
| CLIP ViT-B/16, projected | 0.77835 (1 seed) | — | — | — | — | — |
| DINOv3 ViT-B/16 | 0.77614 ± 0.00056 | 0.7063 | 0.7273 | 0.6760 | 0.7032 | 0.6760 |
| ConvNeXt-B | 0.75623 (1 seed) | 0.6979 | 0.7037 | 0.6502 | 0.6839 | 0.6502 |

Detailed numbers: `backbone_bakeoff.csv`; full run registry: `registry.csv`; per-attribute CSVs and metrics JSON: `results/`. Model revisions and licenses: `model_provenance.json`. The separate C²T checkpoint reference has a different input and training protocol, so it is not part of this backbone ranking.

Per-attribute F1 at threshold 0.5 suggests complementarity: DINOv3 exceeds CLIP on `Backpack` and `Bag`; CLIP exceeds DINOv3 on several clothing colors, `Age-Old`, and `Glasses-Normal`. See `results/clip_vs_dinov3_attribute_delta.csv`. These are one-seed diagnostic deltas; they do not yet justify an ensemble.

## Current GPU timing

RTX A5000 24 GB, one GPU. Full frozen feature extraction (train + val) took roughly 2.4 min DINOv3, 2.4 min ConvNeXt, 3.1 min CLIP, 3.3 min Swin, and 3.4 min SigLIP2. External GPU contention varied. Batch 64 partial fine-tune of the last two vision blocks with AMP was benchmarked for 100 steps: estimated training time per epoch is 114 s CLIP, 120 s SigLIP2, 146 s DINOv3. These estimates exclude validation and model saving; benchmark JSON files are under `benchmarks/`.

CLIP partial fine-tune completed 10 epochs with horizontal flip augmentation and weighted BCE. Total wall time was 1,488 seconds (24 min 48 sec). Best epoch 8: validation mA 0.8639, Inst_F1 0.8356, Avg **0.8498**. Domain Avg: Market1501 0.8159, PA100k 0.8335, PETA 0.8472. A global threshold selected on a stratified half of validation was 0.52 and changed held-half Avg by only +0.0001 compared with 0.5. Since no LODO partial fine-tuning is complete, these pooled scores do not establish domain generalization.

SigLIP2 partial fine-tuning was interrupted by an unrelated GPU workload after epoch 4. Best interim epoch 4 reached Avg 0.8310; the run is incomplete and excluded from completed bake-off comparisons. The process had saved best weights and validation predictions, but not optimizer state, so it cannot resume. The runner now supports `--resume` and saves optimizer/scaler state after each epoch for future runs. At interruption the A5000 had about 19 GiB allocated by another job; the SigLIP2 rerun and DINOv3 partial fine-tune are pending GPU availability. DINOv3 ConvNeXt-B frozen feature extraction support was added but has not yet been run.

## Reproducing the frozen probe

Use `../.conda-upar2027/bin/python` from this repository. Cached features and downloaded weights are local, ignored by Git, and inference requires no network. Example:

```bash
PYTHONDONTWRITEBYTECODE=1 ../.conda-upar2027/bin/python train_linear_probe_2027.py \
  --cache-dir data/feature_cache_clip_full \
  --output-dir experiments/results/lp_clip_example_new_run \
  --epochs 50 --batch-size 2048 --lr 1e-3 --seed 605
```

For LODO, add `--held-out-domain PETA --train-csv ../upar2027_official/data/annotations/task1/train/gt.csv --val-csv ../upar2027_official/data/annotations/task1/val/gt.csv`. Use a new output directory for every run. The CLIP and SigLIP2 stored caches contain pooled visual features; CLIP projected features were derived offline with the trained projection matrix. Dense token caches are deferred until selecting a backbone for attribute-query experiments because caching all tokens for every model is unnecessarily large.
