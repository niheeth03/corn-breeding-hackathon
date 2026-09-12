#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_final_recs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/final_recs_%x.%j.out
#SBATCH --error=logs/final_recs_%x.%j.err

module purge
module load GCC/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs outputs
source "$SCRATCH/corn_env/bin/activate"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u notebooks/03_build_final_recommendations.py \
  --parquet "outputs/merged_master_C${CLUSTER}.parquet" \
  --genomic-dir "Simplified Hackathon Dataset V3/ImputedPopulationsC${CLUSTER}" \
  --cluster "${CLUSTER}" \
  --test-year 2008 \
  --budget 300 \
  --max-per-family 5
