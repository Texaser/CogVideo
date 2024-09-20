run_cmd="CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on $(hostname)"