import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import pandas as pd
from src import genomic_prediction as gp
from src import relatedness_model as rm

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="outputs/merged_master_C1.parquet")
parser.add_argument("--genomic-dir", default="Simplified Hackathon Dataset V3/ImputedPopulationsC1")
parser.add_argument("--cluster", type=int, default=1)
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet)
df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")
pop_table = rm.population_year_target_table(df)
print(f"{len(pop_table)} populations, YEAR range {pop_table['YEAR'].min()}-{pop_table['YEAR'].max()}")

t0 = time.time()
fp_table = rm.build_fingerprint_table(args.genomic_dir, args.cluster, pop_table["pop_num"].tolist())
print(f"Fingerprints built for {len(fp_table)} populations in {time.time()-t0:.1f}s")

sim = rm.similarity_matrix(fp_table)

print("\n=== Tuning bandwidth on 2006 (validation year) ===")
bandwidths = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
results_2006 = []
for bw in bandwidths:
    r = rm.year_forward_relatedness_validation(pop_table, sim, test_year=2006, bandwidth=bw)
    print(f"  bandwidth={bw}: {r}")
    results_2006.append((bw, r["correlation"]))

best_bw = max(results_2006, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]  # nan-safe max
print(f">>> Best bandwidth from 2006: {best_bw}")

print(f"\n=== Internal test on 2007 (bandwidth frozen at {best_bw}, no further tuning) ===")
r2007 = rm.year_forward_relatedness_validation(pop_table, sim, test_year=2007, bandwidth=best_bw)
print(r2007)

print(f"\n=== FINAL real test on 2008 (bandwidth frozen at {best_bw}) ===")
r2008 = rm.year_forward_relatedness_validation(pop_table, sim, test_year=2008, bandwidth=best_bw)
print(r2008)

print("\n=== For comparison, prior models on this same forward-2008 test ===")
print("  Model A (plain ridge on markers):  correlation ~0.09 (C1) / ~0.22 (C2), top20 ~24%/27%")
print("  Model B (mid-parent GCA average):  correlation ~0.005 (C1) / ~0.09 (C2), top20 ~18%/23%")
