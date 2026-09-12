#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=corn_build_master
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/build_master.%j.out
#SBATCH --error=logs/build_master.%j.err

module purge
module load GCCcore/13.2.0 Python/3.11.5

cd "$SCRATCH/ABHackathon"
mkdir -p logs outputs
source "$SCRATCH/corn_env/bin/activate"

# Rebuild both merged master tables on the cluster (uses the full raw data,
# same code that ran locally -- just faster/more memory here).
python src/build_master_dataset.py --mode full --cluster 1
python src/build_master_dataset.py --mode full --cluster 2
