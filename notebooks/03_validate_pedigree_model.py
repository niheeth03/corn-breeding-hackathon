import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from src import genomic_prediction as gp
from src import pedigree_model as pm

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="outputs/merged_master_C1.parquet")
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet, columns=[
    "LINE_UNIQUE_ID", "CROSS", "YEAR", "LOC", "YLD_BE"
])
print(f"Loaded {len(df):,} rows")

df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")
pop_table = pm.population_level_table(df)
print(f"\n{len(pop_table)} populations total")
print(f"Populations with own phenotype data: {pop_table['own_mean'].notna().sum()}")

print("\n=== Model B: leave-population-out pedigree (mid-parent value) validation ===")
result = pm.leave_population_out_pedigree_validation(pop_table)
for k, v in result.items():
    print(f"  {k}: {v}")

print("\n=== For comparison: Model A (within-population genomic) results were ===")
print("  C1: pooled correlation 0.636, median per-family 0.225")
print("  C2: pooled correlation 0.618, median per-family 0.229")
