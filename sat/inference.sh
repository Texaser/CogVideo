#! /bin/bash

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# run_cmd="$environs python sample_video.py --base configs/cogvideox_5b.yaml configs/inference.yaml --seed $RANDOM"
run_cmd="CUDA_VISIBLE_DEVICES=2 $environs python sample_video.py --base configs/cogvideox_5b_lora.yaml configs/inference.yaml --seed 42"


echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"