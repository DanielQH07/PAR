# Local reproduction status

The public UPAR phase 1 data is under `data/phase1/`, the decrypted phase 2 test set is under `data/phase2/`, and the published checkpoint is at `checkpoints/best_model.pth`. The legacy C²T environment used during the inference check is `../.conda-upar-exact` (Python 3.8, Torch 1.13.1+cu117, MMCV Full 1.7.1, timm 0.9.8). For the current research environment and UPAR 2027 data download, use [RUN_RESEARCH.md](RUN_RESEARCH.md), which documents `.venv` and the official organizer download script.

One image was successfully inferred on CPU with:

```bash
head -2 data/phase2/submission_templates_test/task1/predictions.csv > /tmp/upar_smoke_template.csv
UPAR_SKIP_PRETRAINED=1 ../.conda-upar-exact/bin/python infer_upar_test_phase.py \
  --cfg configs/upar_local.yaml \
  --template /tmp/upar_smoke_template.csv \
  --output /tmp/upar_smoke_predictions.csv \
  --batch-size 1 --device cpu
```

`UPAR_SKIP_PRETRAINED=1` skips downloading the Swin and EVA initial weights. The complete published checkpoint is loaded with `strict=True`, so this does not change checkpoint inference. To run all test images when GPU memory is available, omit `--template`, set `--device cuda`, and choose a batch size that fits the GPU. Use a new `--output` path to preserve the repository's `predictions.csv`.

```bash
UPAR_SKIP_PRETRAINED=1 ../.conda-upar-exact/bin/python infer_upar_test_phase.py \
  --cfg configs/upar_local.yaml --batch-size 1 --device cuda \
  --output data/phase2/predictions_reproduced.csv
```

The public phase 1 training CSV contains 97,669 accessible image paths. The phase 2 test set contains 28,095 images. The extracted `data/phase2/annotations/val_gt/val_gt.csv` contains 33,407 accessible validation images. The original `run.sh` referred to a missing `train2.py`, and the training entry point imported a missing local `scheduler` package. Those startup issues are fixed.

One training epoch on a sampled set of 512 train images and 256 validation images completed on the RTX A5000. It used batch size 16, initialized from the published checkpoint, took 28.39 seconds for training, reached validation mA 0.8405, and saved `exp_result/UPAR/swin_b.smoke_epoch1/img_model/best_model.pth`. The configuration is `configs/upar_smoke_epoch1.yaml`; the sampled CSVs and log are under `data/smoke_epoch1/`. This is a training smoke test, not a reproduction of the paper's full training schedule. Batch size 64 ran out of VRAM; batch size 16 peaked around 14.9 GB. Extrapolating the measured train and validation throughput gives roughly 1.5–2 hours for one epoch over all 97,669 training and 33,407 validation images.

The first five CPU predictions differ from the checked in `predictions.csv`; the maximum absolute probability differences per image are 0.485, 0.421, 0.229, 0.107, and 0.182. Repeating the first image in an older environment with Torch 1.10.2 and MMCV 1.3.18 changed probabilities by at most 0.0000006, so the checked in CSV does not match this checkpoint and inference path even with the requested library versions. This comparison does not establish metric reproduction.

The author Docker archive is available as `../upar_hdt.tar`. It is a root filesystem archive, so use `docker import ../upar_hdt.tar upar_hdt:v0` instead of the README's `docker load`. The imported image has Python 3.7, Torch 1.9.0, MMCV 2.1.0, and no `timm`, so it cannot run the repository as supplied. GPU execution in Docker also requires NVIDIA Container Toolkit on this machine (`docker run --gpus all` currently fails).
