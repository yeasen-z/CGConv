#!/usr/bin/env bash
set -euo pipefail

# Override any setting through the environment, for example:
#   EPOCHS=1 NGPU=0 DATA_PATH=./data bash demo.sh
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH="${DATA_PATH:-./data}"
SAVE_PATH="${SAVE_PATH:-./save/cifar10_cgc_resnet56}"
ARCH="${ARCH:-cgc_resnet56}"
EPOCHS="${EPOCHS:-200}"
NGPU="${NGPU:-1}"
GPU_ID="${GPU_ID:-0}"
WORKERS="${WORKERS:-4}"

"${PYTHON_BIN}" cg_train.py \
  --dataset cifar10 \
  --data_path "${DATA_PATH}" \
  --arch "${ARCH}" \
  --save_path "${SAVE_PATH}" \
  --epochs "${EPOCHS}" \
  --learning_rate 0.01 \
  --schedule 80 120 \
  --gammas 0.1 0.1 \
  --train_batch_size 128 \
  --test_batch_size 256 \
  --workers "${WORKERS}" \
  --ngpu "${NGPU}" \
  --gpu_id "${GPU_ID}" \
  --print_freq 100 \
  --decay 0.0005 \
  --momentum 0.9 \
  --cgc
