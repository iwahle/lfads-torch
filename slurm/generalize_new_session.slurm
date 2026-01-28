#!/bin/bash
#SBATCH --job-name=generalize_new_session
#SBATCH --output=slurm/slurm_logs/generalize_new_session-%A.%a.out
#SBATCH --error=slurm/slurm_logs/generalize_new_session-%A.%a.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --array=2-14

module load anacondapy/2023.07-cuda
conda activate lfads-torch

export RAY_TMPDIR=/tmp/$USER-ray
echo $RAY_TMPDIR
mkdir -p $RAY_TMPDIR

echo "Starting generalize_new_session.py SLURM job"
echo "Running on node: $(hostname)"

srun python scripts/run_generalization.py \
    --loo_idx $SLURM_ARRAY_TASK_ID \
    --date 260126
echo "Job finished"
