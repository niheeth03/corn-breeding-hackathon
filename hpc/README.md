# Migrating to TAMU HPRC (Grace)

These steps run in **your own terminal** (or via `!command` in Claude Code) --
they need your NetID + DUO login, which I don't have access to.

## 1. One-time: get on VPN if off-campus
Connect to the TAMU VPN before anything else if you're not on campus network.

## 2. Copy the project to Grace's scratch space
Login nodes cap CPU time at 60 min, so use a Data Transfer Node for the copy
(no time limit, and rsync resumes if interrupted):

```bash
# from your laptop, in the ABHackathon folder
rsync -avz --progress \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  ./ NetID@grace-dtn1.hprc.tamu.edu:/scratch/user/NetID/ABHackathon/
```

This copies your code (src/, hpc/, sample_data/, requirements.txt) **and**
the raw `Simplified Hackathon Dataset V3/` folder. If that folder is large,
expect this to take a while even on the DTN -- check size first:

```bash
du -sh "Simplified Hackathon Dataset V3"
```

## 3. Log in and set up the environment
```bash
ssh NetID@grace.hprc.tamu.edu
cd $SCRATCH/ABHackathon
bash hpc/setup_env.sh
```

## 4. Submit the build job (rebuilds both merged master Parquet files)
```bash
sbatch hpc/build_master_dataset.slurm
squeue -u $USER          # check status
tail -f logs/build_master.*.out   # watch progress
```

## 5. Later: submit the modeling job
Once the genomic prediction / validation code exists (`src/run_pipeline.py`):
```bash
sbatch hpc/run_modeling.slurm
```

## 6. Getting results back to your laptop
```bash
rsync -avz NetID@grace-dtn1.hprc.tamu.edu:/scratch/user/NetID/ABHackathon/outputs/ ./outputs/
```

## Notes
- Everything lives in `$SCRATCH`, never `$HOME` -- `$HOME` has a small file-count
  quota and pip/venv alone can create tens of thousands of files.
- `$SCRATCH` is not backed up and old files are purged periodically -- copy
  final results back to your laptop or `$HOME` before you're done.
- Check `module avail Python` on Grace once logged in -- adjust the module
  version in `setup_env.sh` / the `.slurm` scripts if `Python/3.11.5` isn't
  what's currently installed.
