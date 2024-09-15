#! /bin/bash

environs="CUDA_VISIBLE_DEVICES=3 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python train_video.py --base configs/cogvideox_5b.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"