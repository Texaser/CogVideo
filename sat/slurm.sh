#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodelist=bumblebee.ib
#SBATCH --nodes=1              
#SBATCH --cpus-per-task=16      
#SBATCH --gres=gpu:1 # TODO: ensure this var matches the # GPUs you need
#SBATCH --time=1000:00:00

set -e  # exit script if any command fails

cd /mnt/mir/fan23j/CogVideo/sat


# find this path using `which conda`
CONDA_PATH="/home/fan23j/anaconda3/"
VENV_NAME="cogvideo"

source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${VENV_NAME}"

export OMP_NUM_THREADS=52
export PYTHONWARNINGS="ignore"

# TODO: change to the device(s) you wish to run on
export CUDA_VISIBLE_DEVICES=1

TORCHRUN_OPTIONS=(
    --standalone
    --nproc_per_node=1 # TODO: optionally change this to match # GPUs you plan to run on
)
TRAIN_SCRIPT="train_video.py"
TRAIN_OPTIONS=(
    --wandb
    --base configs/cogvideox_5b_i2v_lora.yaml configs/sft.yaml
)
RUN_CMD=(
    torchrun "${TORCHRUN_OPTIONS[@]}"
    "${TRAIN_SCRIPT}" "${TRAIN_OPTIONS[@]}"
)
"${RUN_CMD[@]}"