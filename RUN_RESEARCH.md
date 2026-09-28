# Local research workflow (UPAR 2027)

This guide documents the current frozen-feature and partial-fine-tuning code in this repository. Run commands from the repository root. Python 3.11 and an NVIDIA GPU are used for the tested setup.

## 1. Create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-research.txt
```

The command above installs the CUDA 13.0 PyTorch wheels used on the RTX A5000 machine. For another CUDA version, use the matching command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Confirm the environment and GPU:

```bash
python -c 'import torch, transformers, timm; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, timm.__version__)'
```

Expected on the tested host: `2.14.0+cu130`, CUDA available, Transformers `4.57.6`, timm `1.0.30`. Activate the environment in each new terminal with `source .venv/bin/activate`. `.venv/` is local and ignored by Git.

## 2. Download and arrange the data

Use the organizer's [UPAR Challenge 2027 data repository](https://github.com/speckean/UPAR-Challenge-2027). It provides annotations and the official `download_datasets.py` script for Market1501, PA-100K, and PETA. The public development data currently has 97,669 train and 33,407 validation images.

From the parent directory of this repository:

```bash
git clone https://github.com/speckean/UPAR-Challenge-2027.git ../upar2027_official
source .venv/bin/activate
python -m pip install gdown
cd ../upar2027_official
python download_datasets.py --data-dir ../upar_challenge/data/phase1
cd ../upar_challenge
```

The official annotation CSVs stay in `../upar2027_official/data/annotations/task1/{train,val}/gt.csv`; image paths in those CSVs resolve under `data/phase1/`. Verify the expected files and one image before starting a long run:

```bash
test -f ../upar2027_official/data/annotations/task1/train/gt.csv
test -f ../upar2027_official/data/annotations/task1/val/gt.csv
test -f data/phase1/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg
```

The organizer's download script downloads public source datasets; it does not provide the hidden test labels. Keep all downloaded data outside Git. For questions about access, licenses, or the source datasets, see the [organizer README](https://github.com/speckean/UPAR-Challenge-2027#dataset-sources).

## 3. Download pretrained model weights

The experiment scripts use local model directories under `weights/`. The CLIP, SigLIP2, Swin, ConvNeXt, and DINOv3 checkpoints and revisions used for the reported runs are listed in [`experiments/model_provenance.json`](experiments/model_provenance.json). Download Hugging Face models with the `hf` CLI, for example:

```bash
hf download openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --local-dir weights/clip_b16
hf download google/siglip2-base-patch16-224 --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 --local-dir weights/siglip2_b16
hf download microsoft/swin-base-patch4-window7-224-in22k --revision 68dc76680a5bf3bdf670669f3025dc9be2e30781 --local-dir weights/swin_b
hf download timm/convnext_base.fb_in22k_ft_in1k model.safetensors --local-dir weights/convnext_b
```

DINOv3 is gated: request access on the [DINOv3 model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), then authenticate interactively with `hf auth login` and download it:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m --revision 5931719e67bbdb9737e363e781fb0c67687896bc --local-dir weights/dinov3_vitb16
```

Do not put a Hugging Face token in a shell command, YAML file, commit, or log. `weights/` and `.env` are ignored by Git. Model IDs and pinned revisions are recorded in `experiments/model_provenance.json`.

## 4. Extract frozen features and train a linear probe

Feature extraction is performed once per backbone. The resulting `.npy` arrays are reused for seeds, evaluation, and LODO probes. This CLIP example extracts all official train and validation images:

```bash
python extract_2027_backbone_features.py \
  --backbone clip_b16 --weights weights/clip_b16 \
  --train-csv ../upar2027_official/data/annotations/task1/train/gt.csv \
  --val-csv ../upar2027_official/data/annotations/task1/val/gt.csv \
  --image-root data/phase1 \
  --output-dir data/feature_cache_clip_full \
  --batch-size 64 --workers 4
```

Train the 50-epoch LayerNorm + Linear(40) head (weighted BCE by default):

```bash
python train_linear_probe_2027.py \
  --cache-dir data/feature_cache_clip_full \
  --output-dir experiments/results/lp_clip_preproj_seed605_50ep \
  --epochs 50 --batch-size 2048 --lr 1e-3 --seed 605
```

The run directory records the best head and `train_summary.json`. Use a new output directory for each run. For held-out-domain evaluation, add the official CSV paths and one `--held-out-domain Market1501`, `PA100k`, or `PETA`; the selected checkpoint is evaluated once on the withheld validation domain:

```bash
python train_linear_probe_2027.py \
  --cache-dir data/feature_cache_clip_full \
  --output-dir experiments/results/lodo_clip_preproj_hold_PETA_50ep \
  --epochs 50 --batch-size 2048 --lr 1e-3 --seed 605 \
  --train-csv ../upar2027_official/data/annotations/task1/train/gt.csv \
  --val-csv ../upar2027_official/data/annotations/task1/val/gt.csv \
  --held-out-domain PETA
```

Use `evaluate_cached_head.py` to compute per-domain/per-attribute metrics from a trained head. `experiments/README.md` describes the current scores and protocol; `experiments/registry.csv` and `experiments/backbone_bakeoff.csv` collect completed runs.

## 5. Partial fine-tuning and resuming

Fine-tune the last two encoder blocks plus the prediction head. Start with a new run folder:

```bash
python train_partial_ft_2027.py \
  --backbone clip_b16 --weights weights/clip_b16 \
  --train-csv ../upar2027_official/data/annotations/task1/train/gt.csv \
  --val-csv ../upar2027_official/data/annotations/task1/val/gt.csv \
  --image-root data/phase1 \
  --output-dir experiments/results/partial_clip_b16_seed605_10ep \
  --epochs 10 --batch-size 64 --val-batch-size 256 --workers 4 --seed 605
```

The script saves `best_model.pth`, predictions, and `last_checkpoint.pth` after each completed epoch. If an interrupted run has a `last_checkpoint.pth`, resume with the same arguments and add `--resume`:

```bash
python train_partial_ft_2027.py \
  --backbone clip_b16 --weights weights/clip_b16 \
  --train-csv ../upar2027_official/data/annotations/task1/train/gt.csv \
  --val-csv ../upar2027_official/data/annotations/task1/val/gt.csv \
  --image-root data/phase1 \
  --output-dir experiments/results/partial_clip_b16_seed605_10ep \
  --epochs 10 --batch-size 64 --val-batch-size 256 --workers 4 --seed 605 --resume
```

## 6. C²T code and data paths

The original C²T pipeline uses the legacy `data/phase1/annotations/phase1/` and `data/phase2/` layout, a published checkpoint, and additional MMCV dependencies. The local config uses repository-relative paths in `configs/upar_local.yaml`. The faster cached-head workflow is documented in [`FEATURE_CACHE.md`](FEATURE_CACHE.md); it starts from the published checkpoint and freezes the backbone, so it is not an independent end-to-end reproduction. The official 2027 frozen-probe scripts above use the organizer's task 1 CSVs directly.

## Local performance reference

Test machine: NVIDIA RTX A5000 24 GB. Full frozen feature extraction for one backbone takes about 2.5–3.5 minutes on an idle GPU. A 50-epoch probe trains from cached vectors and takes much less time than image-level training. Partial fine-tune batch 64 took about 25 minutes per 10-epoch CLIP run. GPU contention can materially increase these times. The latest GPU-specific status and current experiment limits are in [`../GPU_EXECUTION_PLAN.md`](../GPU_EXECUTION_PLAN.md).
