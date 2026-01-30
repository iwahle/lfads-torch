#!/bin/bash
#SBATCH --job-name=prep_loo
#SBATCH --output=slurm/slurm_logs/prepare_loo_data-%A.%a.out
#SBATCH --error=slurm/slurm_logs/prepare_loo_data-%A.%a.err
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-11

# Load conda environment
module load anacondapy/2023.07-cuda
conda activate lfads-torch

echo "========================================"
echo "LOO Data Preparation - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Configuration
DATA_PATH="/jukebox/buschman/Users/Qinpu/Compositionality/ForIman/processed_data/fullTrial_fix_sample"
OUTPUT_DIR="/jukebox/buschman/Users/Iman/lfads/datasets/compositionality"
N_COMPONENTS_KEEP=3

echo ""
echo "Configuration:"
echo "  DATA_PATH: $DATA_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  N_COMPONENTS_KEEP: $N_COMPONENTS_KEEP"
echo "  LOO_IDX: $SLURM_ARRAY_TASK_ID"
echo ""

# Run the data preparation script
srun python scripts/prepare_loo_data.py \
    --loo_idx $SLURM_ARRAY_TASK_ID \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_components_keep $N_COMPONENTS_KEEP

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "========================================"
