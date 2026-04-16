#!/bin/bash

#SBATCH --job-name=rnn_gc_nue_perm_ds2
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --output=%x_%j.stdout
#SBATCH --error=%x-%j.stderr

module purge
module load gcc/12.2.0
module load python/3.10.8-cidwh6y

cd /home/i/intziral/code/RNN-GC
/home/i/intziral/code/project_env/bin/python main_script.py \
    --target $SLURM_ARRAY_TASK_ID \
    --dataset_index 2    