#! /bin/bash
run_cmd="CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 python sample_video.py --base configs/cogvideox_5b_i2v_lora.yaml configs/inference.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"