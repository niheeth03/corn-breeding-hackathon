#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_validate
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/validate.%j.out
#SBATCH --error=logs/validate.%j.err

module purge
module load GCC/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs
source "$SCRATCH/corn_env/bin/activate"

# Without this, NumPy's BLAS backend can spawn threads across every core on
# the node (not just the 8 this job requested), and each thread's working
# memory during the big forward-validation fit (~34k rows x 2911 markers)
# multiplies up -- this is what caused the earlier OOM kill.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python -u notebooks/02_validate_genomic_prediction.py --parquet outputs/merged_master_C1.parquet
