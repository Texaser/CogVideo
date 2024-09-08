#! /bin/bash

# echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# run_cmd="CUDA_VISIBLE_DEVICES=1,2,3,6 torchrun --master_port=25920 --standalone --nproc_per_node=4 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"
run_cmd="CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=7 train_video.py --wandb --base configs/cogvideox_2b_lora.yaml configs/sft.yaml"
# run_cmd="torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_5b.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}
