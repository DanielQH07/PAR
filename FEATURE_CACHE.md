# Training a linear head from cached features

`train_cached_features.py` extracts the 2,048-dimensional output of the published C2T-Net backbone. It loads the published checkpoint, freezes Swin and EVA, and applies the deterministic validation image transform to both training and validation images. The cache consists of NumPy memory-mapped feature and label arrays, plus the initial classifier weights.

```bash
UPAR_SKIP_PRETRAINED=1 ../.conda-upar-exact/bin/python src/train_cached_features.py extract \
  --cfg config/upar_local.yaml --checkpoint checkpoints/best_model.pth \
  --output-dir data/feature_cache_full_amp --batch-size 64 --workers 2 --amp

../.conda-upar-exact/bin/python src/train_cached_features.py train \
  --output-dir data/feature_cache_full_amp --epochs 10 --batch-size 1024 --lr 1e-4
```

The best linear head is `data/feature_cache_full_amp/best_cached_head.pth`. Use it with the original backbone for inference:

```bash
UPAR_SKIP_PRETRAINED=1 ../.conda-upar-exact/bin/python src/infer_upar_test_phase.py \
  --cfg config/upar_local.yaml --checkpoint checkpoints/best_model.pth \
  --head-checkpoint data/feature_cache_full_amp/best_cached_head.pth \
  --batch-size 16 --device cuda --output data/phase2/predictions_cached_head.csv
```

Caching costs one frozen-backbone pass over every image, then head epochs are fast. The corrected 2027 loader uses 97,669 training and 33,407 validation images; these feature vectors take about 1.1 GB on disk in float32. The method changes the training procedure: Swin, EVA, and cross-attention weights no longer update, and cached train images do not receive fresh random crops or flips each epoch. Use the original `train_upar_2024.py` when reproducing end-to-end training.

On this machine, the corrected full extraction took 564.6 seconds including model startup. The published checkpoint has validation mA 0.8599 and Challenge Avg 0.8797. Three new head-only runs selected by Avg reached 0.8805 ± 0.0002 Avg; all three lost some Market1501 Avg. The published checkpoint was used as the starting point; this run fine-tunes only its linear head.
