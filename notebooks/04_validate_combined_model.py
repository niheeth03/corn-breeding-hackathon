import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from src import genomic_prediction as gp
from src import combined_model as cm

FROZEN_ALPHA = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="outputs/merged_master_C1.parquet")
parser.add_argument("--test-year", type=int, default=2008)
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet)
print(f"Loaded {len(df):,} rows")

marker_cols = gp.marker_columns(df)
df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")

print(f"\n=== Model A vs Model B vs Combined, test_year={args.test_year} ===")
a, b, combined = cm.combined_ab_forward_validation(df, marker_cols, test_year=args.test_year, alpha=FROZEN_ALPHA)
print("Model A only (DNA, pooled ridge):     ", a)
print("Model B only (parent GCA/pedigree):   ", b)
print("Combined (A within-family + B family):", combined)
