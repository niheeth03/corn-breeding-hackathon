#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_combined
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/combined_%x.%j.out
#SBATCH --error=logs/combined_%x.%j.err

module purge
module load GCC/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs
source "$SCRATCH/corn_env/bin/activate"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u notebooks/04_validate_combined_model.py --parquet "$PARQUET_FILE" --test-year 2008
