#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_upar_2024.py --cfg "${UPAR_CFG:-./configs/upar.yaml}"
