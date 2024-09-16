#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodelist=bumblebee.ib
#SBATCH --nodes=1              
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

set -e  # exit script if any command fails
cd /mnt/mir/levlevi/CogVideo/sat

CONDA_PATH="/playpen-storage/levlevi/anaconda3"
VENV_NAME="cogvideo"

source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${VENV_NAME}"

NUM_PROCS=1
TRAIN_SCRIPT="train_video.py"
CONFIG_FILES="configs/cogvideox_5b_lora.yaml configs/sft.yaml"
ADDITIONAL_ARGS="--wandb"

export OMP_NUM_THREADS=52
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=2,3

TORCHRUN_OPTIONS=(
    --standalone
    --nproc_per_node=2
)
TRAIN_SCRIPT="train_video.py"
TRAIN_OPTIONS=(
    --wandb
    --base configs/cogvideox_5b_lora.yaml configs/sft.yaml
)
RUN_CMD=(
    torchrun "${TORCHRUN_OPTIONS[@]}"
    "${TRAIN_SCRIPT}" "${TRAIN_OPTIONS[@]}"
)
"${RUN_CMD[@]}"