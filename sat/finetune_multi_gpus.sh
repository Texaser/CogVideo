#!/bin/bash

export OMP_NUM_THREADS=52
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0

TORCHRUN_OPTIONS=(
    --standalone
    --nproc_per_node=1
)
TRAIN_SCRIPT="train_video.py"
TRAIN_OPTIONS=(
    --wandb
    --base configs/cogvideox_2b_lora.yaml configs/sft.yaml
)
RUN_CMD=(
    torchrun "${TORCHRUN_OPTIONS[@]}"
    "${TRAIN_SCRIPT}" "${TRAIN_OPTIONS[@]}"
)
"${RUN_CMD[@]}"