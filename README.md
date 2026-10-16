# UPAR 2027 Track 1: pedestrian attribute recognition under domain shift

Entry for the [UPAR Challenge 2027, Track 1](https://www.codabench.org/competitions/18180/). The model predicts 40
binary attributes per pedestrian crop. It trains on Market1501, PA100k and PETA and is tested on unseen domains
(MEVID, synthetic). The submission is code that runs on an A10 GPU within 300 s.

## Architecture

```
image ──► T2   MetaCLIP ViT-B/16, 256×128 ──► 40 logits ─┐
      ──► PE-B PE-Core B/16,     256×128 ──► 40 logits ─┼─► calibrate ─► sigmoid ─► per-attribute weighted mean ─► + bias ─► > 0.5
      ──► PE-L PE-Core L/14,     280×140 ──► 40 logits ─┘
```

| Member | Weight on colour attributes (24) | Weight on other attributes (16) |
|---|---:|---:|
| T2 (MetaCLIP B/16, soup of 3 seeds, train+val) | 0.5 | 0.5 |
| PE-B (PE-Core B/16, train) | 0.5 | 0 |
| PE-L (PE-Core L/14, train) | 0 | 0.5 |

- **Each member:**
  - an image encoder initialised from generic image–text pretraining;
  - the pooled embedding goes through LayerNorm + Linear(40).
- **Training:**
  - 1 linear-probe epoch, then full fine-tuning with layer-wise lr decay;
  - AdamW, EMA 0.999;
  - BCE with pos_weight ((N−P)/P)^0.5, label smoothing 0.1, unannotated (domain, attribute) pairs masked;
  - domain-balanced sampling;
  - augmentation: crop, flip, low-res, blur, brightness/contrast, random erasing.
- **Calibration:** each member's logits are shifted by 0.1 × (mean logit on 4 fixed content-free images − prior
  logit).
- **Bias:** a fixed +0.005 probability is added to Age-Adult and LowerBody-Length-Short.
- **Selection:** leave-one-domain-out (LODO). Train on two source datasets, score the third, for three folds. Keep a
  change only if Challenge_Avg rises and instance recall and colour mA do not fall on every fold (`src/lodo_gate.py`).
  Test data and leaderboard results are never used for selection.

## Code

```
src/
├── train.py              # training (--arch b16 | l14 | siglip | pe | pel), held-out scoring -> probe.npz
├── prepare.py            # fixed held-out scorer (calibration + Codabench metric)
├── job.sh                # one recipe on the 3 LODO folds
├── lodo_gate.py          # compare two ensemble layouts on the LODO probes
├── export_pe.py          # PE checkpoint -> TorchScript + calibration offsets
├── v8_pipeline.sh        # T2: 3 seeds + weight soup + single-member package
├── package_submission.py # single-member package
├── build_ensemble.py     # T2 package + PE members -> submission zip
├── clip_vit.py           # CLIP vision tower shared by training and the submission
└── download_weights.py   # pretrained backbones
submission/task1/         # run.py (inference, time budget), metadata.yaml, assets/
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-research.txt
git clone https://github.com/speckean/UPAR-Challenge-2027.git ../upar2027_official
(cd ../upar2027_official && python download_datasets.py --data-dir "$OLDPWD/data/phase1")
ln -s ../upar2027_official/data/annotations/task1 ann && ln -s data/phase1 images
python src/download_weights.py metaclip pe pel
```

## Build the submission

```bash
export PYTHONPATH=src
COMMON="--train-csv ann/train/gt.csv --val-csv ann/val/gt.csv --image-root images --batch-size 64 --workers 2"

WEIGHTS=weights_own/metaclip_b16 NAME=t2 CC=0.1 \
  V8_ARGS="--unfreeze 12 --llrd 0.8 --ft-samples 180000 --exact-samples --train-on-val" bash src/v8_pipeline.sh
python src/train.py --arch pe  --weights weights_own/pe_core_b16 --unfreeze 12 --llrd 0.8 --ft-samples 180000 \
  --seed 606 $COMMON --output-dir runs/full_pe606
python src/train.py --arch pel --weights weights_own/pe_core_l14 --size 280 140 --unfreeze 24 --llrd 0.85 \
  --ft-samples 360000 --seed 606 $COMMON --output-dir runs/full_pel280

UPAR_EXPORT_RUN=runs/full_pe606  UPAR_EXPORT_OUT=exports/pe606  UPAR_EXPORT_FILE=pe606.pt  python src/export_pe.py
UPAR_EXPORT_RUN=runs/full_pel280 UPAR_EXPORT_OUT=exports/pel280 UPAR_EXPORT_FILE=pel280.pt python src/export_pe.py

python src/build_ensemble.py --base builds/ensemble_t2 --out builds/ensemble_v24 \
  --pe exports/pel280/pel280.pt:other --pe exports/pe606/pe606.pt:colour \
  --bias Age-Adult=0.005 --bias LowerBody-Length-Short=0.005
```

Upload `builds/ensemble_v24.zip` as is. Hardware: one 24 GB GPU (PE-L 280×140 needs ~22 GB at batch 64).

## Evaluate a change

```bash
bash src/job.sh MYRUN --arch pel --weights weights_own/pe_core_l14 --size 280 140 --unfreeze 24 --llrd 0.85 \
  --ft-samples 360000 --seed 606 --batch-size 64 --workers 2
python src/lodo_gate.py --expert t2=runs/lodo/T2 --expert pe606=runs/lodo/PE606 --expert new=runs/lodo/MYRUN \
  --baseline "colour=t2+pe606 other=t2+pe606" --candidate "colour=t2+pe606 other=t2+new"
```
