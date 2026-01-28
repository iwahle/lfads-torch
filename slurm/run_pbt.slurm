#!/bin/bash
#SBATCH --job-name=run_pbt
#SBATCH --output=slurm/slurm_logs/run_pbt-%A.%a.out
#SBATCH --error=slurm/slurm_logs/run_pbt-%A.%a.err
#SBATCH --time=05:55:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --array=0-14

module load anacondapy/2023.07-cuda
conda activate lfads-torch

export RAY_TMPDIR=/tmp/$USER-ray
echo $RAY_TMPDIR
mkdir -p $RAY_TMPDIR

echo "Starting run_pbt.py SLURM job"
echo "Running on node: $(hostname)"

srun python scripts/run_pbt.py --loo $SLURM_ARRAY_TASK_ID

echo "Job finished"
