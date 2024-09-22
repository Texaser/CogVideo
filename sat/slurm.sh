#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodelist=bumblebee.ib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00

# set -e  # exit script if any command fails
cd /mnt/mir/levlevi/CogVideo/sat

# find this path using `which conda`
# CONDA_PATH="/playpen-storage/levlevi/anaconda3/condabin/conda"
VENV_NAME="cogvideo"

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate "${VENV_NAME}"

export OMP_NUM_THREADS=52
export PYTHONWARNINGS="ignore"

# TODO: change to the device(s) you wish to run on
export CUDA_VISIBLE_DEVICES=2

# TODO: optionally change --nproc_per_node to match # GPUs you plan to run on
TORCHRUN_OPTIONS=(
    --standalone
    --nproc_per_node=1
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

echo "${RUN_CMD[@]}"
torchrun "${TORCHRUN_OPTIONS[@]}" "${TRAIN_SCRIPT}" "${TRAIN_OPTIONS[@]}"