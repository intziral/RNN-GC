#!/bin/bash
#SBATCH --job-name=combine_results_ds2
#SBATCH --partition=batch
#SBATCH --time=5:00
#SBATCH --nodes=1
#SBATCH --output=%x_%j.stdout
#SBATCH --error=%x-%j.stderr

module purge
module load gcc/12.2.0
module load python/3.10.8-cidwh6y

cd /home/i/intziral/code/RNN-GC
/home/i/intziral/code/project_env/bin/python process_results.py \
    --dataset_index 2   