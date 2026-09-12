import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from src import genomic_prediction as gp
from src import validation as val
from src import relatedness_model as rm

FROZEN_ALPHA = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default="outputs/merged_master_C1.parquet")
parser.add_argument("--genomic-dir", default="Simplified Hackathon Dataset V3/ImputedPopulationsC1")
parser.add_argument("--cluster", type=int, default=1)
parser.add_argument("--bandwidths", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet)
marker_cols = gp.marker_columns(df)
df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")

# --- Model A: score on 2006 and 2007 ---
modelA_2006, _, _, _, _ = val.forward_year_validation(df, marker_cols, test_year=2006, alpha=FROZEN_ALPHA)
modelA_2007, _, _, _, _ = val.forward_year_validation(df, marker_cols, test_year=2007, alpha=FROZEN_ALPHA)
modelA_avg = (modelA_2006["correlation"] + modelA_2007["correlation"]) / 2
print(f"\nModel A:  2006={modelA_2006['correlation']:.4f}  2007={modelA_2007['correlation']:.4f}  avg={modelA_avg:.4f}")

# --- Relatedness model: tune bandwidth on 2006, then score both 2006 and 2007 at that bandwidth ---
pop_table = rm.population_year_target_table(df)
fp_table = rm.build_fingerprint_table(args.genomic_dir, args.cluster, pop_table["pop_num"].tolist())
sim = rm.similarity_matrix(fp_table)

tuning = [(bw, rm.year_forward_relatedness_validation(pop_table, sim, 2006, bw)["correlation"]) for bw in args.bandwidths]
for bw, c in tuning:
    print(f"    bandwidth={bw}: 2006 correlation={c:.4f}")
best_bw = max(tuning, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]

rel_2006 = rm.year_forward_relatedness_validation(pop_table, sim, 2006, best_bw)
rel_2007 = rm.year_forward_relatedness_validation(pop_table, sim, 2007, best_bw)
rel_avg = (rel_2006["correlation"] + rel_2007["correlation"]) / 2
print(f"\nRelatedness (bandwidth={best_bw}):  2006={rel_2006['correlation']:.4f}  2007={rel_2007['correlation']:.4f}  avg={rel_avg:.4f}")

# --- Decide winner using ONLY 2006+2007, before looking at 2008 ---
winner = "relatedness" if rel_avg > modelA_avg else "modelA"
print(f"\n>>> WINNER (decided from 2006+2007 only): {winner}")

# --- Apply winner once to the real 2008 test ---
if winner == "modelA":
    final, _, _, _, _ = val.forward_year_validation(df, marker_cols, test_year=2008, alpha=FROZEN_ALPHA)
else:
    final = rm.year_forward_relatedness_validation(pop_table, sim, 2008, best_bw)

print(f"\n=== FINAL 2008 result using selected model ({winner}) ===")
print(final)

# --- For honesty: also report what the OTHER model would have gotten on 2008 ---
print("\n=== For comparison: what the non-selected model gets on 2008 (not used for the decision) ===")
if winner == "modelA":
    alt = rm.year_forward_relatedness_validation(pop_table, sim, 2008, best_bw)
    print("Relatedness (not selected):", alt)
else:
    alt, _, _, _, _ = val.forward_year_validation(df, marker_cols, test_year=2008, alpha=FROZEN_ALPHA)
    print("Model A (not selected):", alt)
