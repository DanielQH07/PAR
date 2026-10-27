# UPAR 2027 Track 1: pedestrian attribute recognition under domain shift

Entry for the [UPAR Challenge 2027, Track 1](https://www.codabench.org/competitions/18180/). The model predicts 40
binary attributes per pedestrian crop. It trains on Market1501, PA100k and PETA and is tested on unseen domains
(MEVID, synthetic). The submission is code that runs on an A10 GPU within 300 s.

## Architecture

The current best submission is `178` (Codabench Challenge_Avg 0.8006).

```
image ──► MetaCLIP ViT-B/16, 288×144 ──► 40 logits ─┐
      ──► PE-B PE-Core B/16, 256×128 ──► 40 logits ─┼─► calibrate ─► sigmoid ─► per-attribute weighted mean ─► + bias ─► > 0.5
      ──► PE-L PE-Core L/14, 336×168 ──► 40 logits ─┘
```

| Member | Training data | Weights | Colour attributes (24) | Other attributes (16) |
|---|---|---|---:|---:|
| MetaCLIP B/16 288×144 | train+val | uniform soup of 3 seeds | 1/3 | 1/2 |
| PE-B PE-Core B/16 256×128 | train | uniform soup of 3 seeds | 1/3 | 0 |
| PE-L PE-Core L/14 336×168 | train | soup of 3 seeds, WiSE-FT α 0.7 towards the pre-trained weights | 1/3 | 1/2 |

- **Each member:**
  - an image encoder initialised from generic image–text pretraining;
  - the pooled embedding goes through LayerNorm + Linear(40).
- **Training:**
  - 1 linear-probe epoch, then full fine-tuning with layer-wise lr decay (0.8 for B/16, 0.85 for L/14);
  - AdamW, EMA 0.999;
  - BCE with pos_weight ((N−P)/P)^0.5, label smoothing 0.1, unannotated (domain, attribute) pairs masked;
  - domain-balanced sampling;
  - augmentation: crop, flip, low-res, blur, brightness/contrast, random erasing;
  - MEVID-like degradation (motion blur + 64–128 px + JPEG q30–50) with p 0.4 for MetaCLIP and PE-L;
  - PE-L also uses stronger degradation: low-res p 0.5 and JPEG p 0.3.
- **Calibration:** each member's logits are shifted by 0.1 × (mean logit on 4 fixed content-free images − prior
  logit).
- **Bias:** +0.10 probability on UpperBody-Color-Other and LowerBody-Color-Other, −0.05 on Age-Adult.
- **Selection:** leave-one-domain-out (LODO). Train on two source datasets, score the third, for three folds. Keep a
  change only if Challenge_Avg rises and instance recall and colour mA do not fall on every fold (`src/lodo_gate.py`).
  Test images and labels are never used for training or calibration. Later rounds train on train+val, so val no
  longer ranks submissions; the Color-Other / Age-Adult biases, the WiSE α and the member choice were then picked
  from the Codabench leaderboard (3 submissions per day), which risks overfitting the public test set.

All submissions and their scores are listed in `result.md`.

## Code

```
src/
├── train.py              # training (--arch b16 | l14 | siglip | pe | pel), held-out scoring -> probe.npz
├── prepare.py            # fixed held-out scorer (calibration + Codabench metric)
├── job.sh                # one recipe on the 3 LODO folds
├── lodo_gate.py          # compare two ensemble layouts on the LODO probes
├── soup.py               # uniform weight soup of several runs
├── wise_from_run.py      # soup + WiSE-FT of PE runs
├── export_pe.py          # PE checkpoint -> TorchScript + calibration offsets
├── v8_pipeline.sh        # T2: 3 seeds + weight soup + single-member package (+ smoke checks)
├── package_submission.py # single-member package
├── build_ensemble.py     # base package + PE members -> submission zip
├── tweak_zip.py          # re-weight members / add biases on a built zip
├── build_submission.sh   # final 3-member layout (build_ensemble.py + tweak_zip.py)
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

Recipe of `178`. Seeds 605, 606 and 607 for every member.

```bash
export PYTHONPATH=src
COMMON="--train-csv ann/train/gt.csv --val-csv ann/val/gt.csv --image-root images --workers 4"

# MetaCLIP B/16 288×144, train+val, MEVID-like degradation -> soup -> base package
for s in 605 606 607; do
  python src/train.py --weights weights_own/metaclip_b16 --unfreeze 12 --llrd 0.8 --ft-samples 180000 \
    --exact-samples --train-on-val --size 288 144 --mevid-prob 0.4 --seed $s $COMMON --output-dir runs/mc288mevid_s$s
done
python src/soup.py runs/mc288mevidsoup3 runs/mc288mevid_s605 runs/mc288mevid_s606 runs/mc288mevid_s607
python src/package_submission.py --weights weights_own/metaclip_b16 --train-csv ann/train/gt.csv \
  --val-csv ann/val/gt.csv --cc 0.1 --member dg:runs/mc288mevidsoup3 --out builds/ensemble_mc288mevid

# PE-B 256×128 -> soup (α 1.0 = plain soup) -> TorchScript
for s in 605 606 607; do
  python src/train.py --arch pe --weights weights_own/pe_core_b16 --unfreeze 12 --llrd 0.8 --ft-samples 180000 \
    --batch-size 64 --seed $s $COMMON --output-dir runs/full_pe$s
done
python src/wise_from_run.py runs/pebsoup3 1.0 weights_own/pe_core_b16 runs/full_pe605 runs/full_pe606 runs/full_pe607
UPAR_EXPORT_RUN=runs/pebsoup3 UPAR_EXPORT_OUT=exports/pebsoup3 UPAR_EXPORT_FILE=pebsoup3.pt python src/export_pe.py

# PE-L 336×168, MEVID-like + stronger degradation -> soup + WiSE-FT α 0.7 -> TorchScript
for s in 605 606 607; do
  python src/train.py --arch pel --weights weights_own/pe_core_l14 --unfreeze 24 --llrd 0.85 --batch-size 32 \
    --grad-ckpt --size 336 168 --ft-samples 360000 --mevid-prob 0.4 --lowres-prob 0.5 --jpeg-prob 0.3 \
    --seed $s $COMMON --output-dir runs/pel336mevidstr_s$s
done
python src/wise_from_run.py runs/pel336mevidstrw7 0.7 weights_own/pe_core_l14 \
  runs/pel336mevidstr_s605 runs/pel336mevidstr_s606 runs/pel336mevidstr_s607
UPAR_EXPORT_RUN=runs/pel336mevidstrw7 UPAR_EXPORT_OUT=exports/pel336mevidstrw7 UPAR_EXPORT_FILE=pel336mevidstrw7.pt \
  python src/export_pe.py

# 3-member zip: weights per attribute group + Color-Other / Age-Adult biases
bash src/build_submission.sh builds/ensemble_mc288mevid.zip exports/pel336mevidstrw7/pel336mevidstrw7.pt \
  178 mc288mevid-pebcol-pel336mevidstrw7-colother10-adult05
```

The zip is saved as `{id}_{name}.zip` by `src/save_submission.py` (directory set by `UPAR_SUBMIT_DIR`); upload it as
is. Hardware: 16 GB GPU is enough (PE-L at batch 32 with gradient checkpointing). Training time is about 5 h on one
RTX 5080 (PE-L ≈ 90 min per seed).

## Evaluate a change

```bash
bash src/job.sh MYRUN --arch pel --weights weights_own/pe_core_l14 --size 280 140 --unfreeze 24 --llrd 0.85 \
  --ft-samples 360000 --seed 606 --batch-size 64 --workers 2
python src/lodo_gate.py --expert t2=runs/lodo/T2 --expert pe606=runs/lodo/PE606 --expert new=runs/lodo/MYRUN \
  --baseline "colour=t2+pe606 other=t2+pe606" --candidate "colour=t2+pe606 other=t2+new"
```
