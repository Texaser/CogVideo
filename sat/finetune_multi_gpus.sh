#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --master_port=25920 --nproc_per_node=1 train_video.py \
    --wandb \
    --base configs/cogvideox_2b_lora.yaml \
    configs/sft.yaml