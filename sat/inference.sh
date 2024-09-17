#! /bin/bash

environs="CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video.py --base configs/cogvideox_5b_lora.yaml configs/inference.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"