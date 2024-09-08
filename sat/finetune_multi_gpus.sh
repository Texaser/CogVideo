#!/bin/bash

# Set the path to your Anaconda environment
CONDA_ENV_PATH="/mnt/mir/fan23j/anaconda3/envs/cogvideo"

# Add the Anaconda environment's bin directory to the PATH
export PATH="$CONDA_ENV_PATH/bin:$PATH"

# Specify the full path to the Python interpreter
PYTHON_PATH="$CONDA_ENV_PATH/bin/python"

# Set up the run command using the specified Python interpreter
run_cmd="CUDA_VISIBLE_DEVICES=1,2,3 $PYTHON_PATH -m torch.distributed.run --standalone --nproc_per_node=3 train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on $(hostname)"