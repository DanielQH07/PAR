# UPAR 2027 Track 1: pedestrian attribute recognition under domain shift

Code for our entry to the [UPAR Challenge 2027, Track 1](https://www.codabench.org/competitions/18180/) (WACV 2027 RWS
workshop). The task is to predict 40 binary attributes for each pedestrian crop. The model trains on Market1501, PA100k
and PETA and is tested on two unseen domains, **MEVID** (surveillance video) and a **synthetic** set. Score:
`Challenge_Avg` = harmonic mean of mA and instance F1, averaged over the two test sets. The submission is code that
runs on a **CPU-only** scorer within **3600 s** for about 29k images.

## How we approached it

1. **Measure what the leaderboard measures.** The test domains are hidden, so every idea is scored with
   leave-one-domain-out (LODO): train on two source datasets, test on the third, three folds. Scoring runs the exact
   submission path (one model, contextual calibration, Codabench metric). In-domain validation over-estimates by
   about 0.08 and ranks ideas wrongly. The Market1501 fold (re-ID crops, many cameras) tracks MEVID best.
2. **Fit the CPU budget first.** On the scorer, one ViT-B/16 at 256x128 takes 1900-3200 s, so the submission is
   always one ViT-B: no ensemble, no TTA, no bigger input, fp32. Everything else (teachers, extra losses) is
   training-only.
3. **Pick the backbone by LODO.** Fine-tuning web-scale image-text models generalises best. MetaCLIP B/16 beat
   OpenAI CLIP, DFN-2B, SigLIP 2 (FixRes and NaFlex), CLIP ViT-L/14 and PE-Core B/16.
4. **Find the right amount of fine-tuning.** We scored the held-out fold after every epoch and found that it peaks
   after about 2 epochs (~180k images) and then falls, while in-domain validation keeps rising. Longer training fits
   the source datasets' labelling conventions (for example, faded jeans called blue in one dataset and grey in
   another), and this did not change with LoRA or noisy-label losses. We fix the length (`--ft-samples`) instead of
   choosing an epoch on validation. This gave the largest single gain on the leaderboard.
5. **Diagnose before adding things.** Error decomposition on held-out folds (`src/error_anatomy.py`)
   showed that we lose to the top team on instance precision, not on mA. Most errors come from colour naming
   conflicts between datasets and from small accessories. More than 60 ideas were tested on LODO (heads, losses,
   layer schedules, LoRA, augmentation, crowd layers, feature distillation). Most traded mA for F1 one-for-one, so
   the harmonic mean did not move. Logit distillation from an ensemble of two different backbones (MetaCLIP + PE-Core)
   raised precision without losing mA.

Final recipe: MetaCLIP ViT-B/16, input 256x128 with aspect-preserving padding, full fine-tune with layer-wise lr
decay 0.8, EMA 0.999, BCE with pos_weight (neg/pos)^0.5 and label smoothing 0.1, domain-balanced sampling,
unannotated (domain, attribute) pairs masked, 2 epochs, weight soup of 3 seeds, contextual calibration (λ 0.1) at
inference.

## Quick start

```bash
# 1. environment + data: RUN_RESEARCH.md sections 1-2, then point the scripts at the data
ln -s data/annotations/task1 ann && ln -s data/phase1 images
python src/download_weights.py                       # MetaCLIP, OpenAI CLIP, PE-Core

# 2. score a recipe on the 3 LODO folds (one GPU, ~45 min)
bash src/job.sh MYRUN --weights weights_own/metaclip_b16 --unfreeze 12 --llrd 0.8 --ft-samples 180000

# 3. build a submission zip (3 seeds + soup + packaging + Docker CPU check) -> builds/ensemble_final.zip
WEIGHTS=weights_own/metaclip_b16 NAME=final \
  V8_ARGS="--unfreeze 12 --llrd 0.8 --ft-samples 180000 --exact-samples --train-on-val" bash src/v8_pipeline.sh
```

### Where things are

```
src/
├── download_weights.py     # MetaCLIP B/16, OpenAI CLIP B/16, PE-Core B/16 (~1.3 GB) -> weights/, weights_own/
├── train.py                # one training run; every option, including those that did not help (--help)
├── prepare.py              # fixed held-out scorer: single model, contextual calibration, Codabench metric
├── job.sh                  # one recipe on the 3 LODO folds -> logs/<RUN>.<fold>.log, runs/lodo/<RUN>/<fold>/probe.npz
├── v8_pipeline.sh          # 3 seeds on all data + weight soup + packaging + Docker CPU check -> builds/ensemble_<NAME>.zip
├── package_submission.py   # builds the Codabench zip
├── clip_vit.py             # dependency-free CLIP / SigLIP vision tower, shared by training and the submission
├── robust2.py, metric_split.py, error_anatomy.py, layer_drift.py   # analysis of held-out probes
└── ...                     # earlier 2027 baselines and legacy C2T-Net code
submission/task1/
├── run.py                  # submission runtime: batching, time budget, calibration
├── metadata.yaml
└── assets/attribute_prior.json
```

Scripts call `python`; set `PY=/path/to/python` for another interpreter. Upload `builds/ensemble_<NAME>.zip` to
Codabench as is (torch, numpy and Pillow only).

Comparing runs on the held-out probes (`robust2` drops attributes with < 20 source or < 50 held-out positives):

```bash
PYTHONPATH=src python src/robust2.py                         # edit `tags` inside: robust and robust2 per fold
PYTHONPATH=src python src/metric_split.py RUN_A RUN_B         # mA vs instance F1 / precision / recall
PYTHONPATH=src python src/error_anatomy.py RUN Market1501     # per-attribute FP/FN and colour confusions
```

Tried on LODO without a gain: SigLIP 2 (FixRes, NaFlex), DFN, ViT-L, PE-Core alone, LoRA, reverse layer-wise lr,
frozen top blocks, focal loss, lower pos_weight, crowd layer, colour-group CE, ELR, Lipsum-FT, two-view consistency,
MixStyle, intermediate CLS fusion, attribute and query heads, same-person consistency, token-level feature
distillation, longer training (with or without val data).

## Legacy: C²T-Net (UPAR 2024)

This repository started from C²T-Net (Swin + EVA ViT, a UPAR 2024 Track 1 entry). Its training and inference
(`run.sh`, `src/infer_upar*.py`) are kept for reference.

```
@InProceedings{Bui_2024_WACV,
    author    = {Bui, Doanh C. and Le, Thinh V. and Ngo, Ba Hung},
    title     = {C2T-Net: Channel-Aware Cross-Fused Transformer-Style Networks for Pedestrian Attribute Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    year      = {2024},
    pages     = {351-358}
}
```
