#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_modeling
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/modeling.%j.out
#SBATCH --error=logs/modeling.%j.err

module purge
module load GCCcore/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs outputs
source "$SCRATCH/corn_env/bin/activate"

# Placeholder -- points at the genomic prediction + validation pipeline once
# src/genomic_prediction.py and src/validation.py exist (Phase 3-4).
# Naive vs leave-population-out CV and the 2007->2008 holdout are both CPU/
# memory heavy across ~1000 populations, which is exactly why this runs here
# instead of on a laptop.
python src/run_pipeline.py --cluster 1
python src/run_pipeline.py --cluster 2
