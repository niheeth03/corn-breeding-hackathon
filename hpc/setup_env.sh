#!/bin/bash
# Run this ONCE on a Grace login node after your code + data have been
# rsynced over. Builds the Python environment in $SCRATCH (never $HOME --
# HOME has a small file-count quota and pip/venv create thousands of files).
#
# Usage:
#   ssh NetID@grace.hprc.tamu.edu
#   cd $SCRATCH/ABHackathon
#   bash hpc/setup_env.sh

set -euo pipefail

module purge
module load GCCcore/13.2.0 Python/3.11.5   # adjust version if `module avail Python` shows a different one

cd "$SCRATCH/ABHackathon"

python3 -m venv "$SCRATCH/corn_env"
source "$SCRATCH/corn_env/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready at $SCRATCH/corn_env"
echo "Activate it in job scripts with: source \$SCRATCH/corn_env/bin/activate"
