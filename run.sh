#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python src/train_upar_2024.py --cfg "${UPAR_CFG:-./config/upar.yaml}"
