#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_pedigree
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/pedigree_%x.%j.out
#SBATCH --error=logs/pedigree_%x.%j.err

module purge
module load GCC/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs
source "$SCRATCH/corn_env/bin/activate"

python -u notebooks/03_validate_pedigree_model.py --parquet "$PARQUET_FILE"
