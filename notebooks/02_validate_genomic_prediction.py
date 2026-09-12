import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from src import genomic_prediction as gp
from src import validation as val

FROZEN_ALPHA = 1000  # confirmed independently by tuning (this run) and by senior's own analysis

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="outputs/merged_master_C1.parquet")
parser.add_argument("--retune", action="store_true", help="Re-run the alpha grid search instead of using the frozen value")
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet)
print(f"Loaded {len(df):,} rows")

marker_cols = gp.marker_columns(df)
print(f"{len(marker_cols)} marker columns")

df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")

X, y = gp.build_line_dataset(df, marker_cols, "YLD_BE_ADJ")
print(f"\nLine-level dataset: {len(X)} genotyped lines with at least one yield record")

if args.retune:
    print("\n=== Tuning ridge alpha via within-population CV ===")
    best_alpha, alpha_results = val.tune_alpha_within_population(X, y)
    print(f">>> Best alpha: {best_alpha}")
else:
    best_alpha = FROZEN_ALPHA
    print(f"\n=== Using frozen alpha={best_alpha} (skip --retune to re-run grid search) ===")

print("\n=== Naive random K-fold CV (leaky -- reference only, not a candidate result) ===")
naive_metrics, _ = val.naive_random_cv(X, y, alpha=best_alpha)
print(naive_metrics)

print("\n=== Leave-population-out CV (cross-family -- reference only, not a candidate result) ===")
lpo_metrics, _ = val.leave_population_out_cv(X, y, alpha=best_alpha)
print(lpo_metrics)

print("\n=== Within-population CV (PRIMARY metric #1: within-family deployable scenario) ===")
wp_metrics, wp_preds, per_pop = val.within_population_cv(X, y, alpha=best_alpha)
valid = wp_preds.notna()
wp_metrics["top20pct_recovery"] = val.top_k_recovery(y[valid], wp_preds[valid], frac=0.2)
print(wp_metrics)

print(f"\n>>> Naive CV correlation (reference only):   {naive_metrics['correlation']:.3f}")
print(f">>> Leave-population-out (reference only):   {lpo_metrics['correlation']:.3f}")
print(f">>> Within-population correlation:            {wp_metrics['correlation']:.3f}  <- PRIMARY")
print(f">>> Within-population top-20% recovery:       {wp_metrics['top20pct_recovery']}")

print("\n=== PRIMARY metric #2: Year-forward validation ===")
print("    2006 and 2007 = internal rehearsal (years we already fully know, used to build confidence)")
print("    2008 = the real final check (matches the hackathon's actual decision point)")
for test_year in [2006, 2007, 2008]:
    if df["YEAR"].max() < test_year or df["YEAR"].min() > test_year - 1:
        print(f"\n--- test_year={test_year}: skipped, insufficient year range in this data ---")
        continue
    print(f"\n--- test_year={test_year} (train through {test_year - 1}) ---")
    fwd, null_b, pheno_b, preds, y_test = val.forward_year_validation(df, marker_cols, test_year=test_year, alpha=best_alpha)
    print("Genomic model:      ", fwd)
    print("Null baseline:      ", null_b)
    print("Phenotypic baseline:", pheno_b)

if df["YEAR"].max() >= 2008 and df["YEAR"].min() <= 2007:
    print("\n=== Does genomics beat farm means on RAW yield? (2008 holdout) ===")
    farm_only, combined = val.forward_raw_yield_vs_farm_mean(df, marker_cols, alpha=best_alpha)
    print("Farm-mean only:              ", farm_only)
    print("Farm-mean + genomic deviation:", combined)
