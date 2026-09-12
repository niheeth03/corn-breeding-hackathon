import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from src import genomic_prediction as gp
from src import final_recommendation as fr

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", required=True)
parser.add_argument("--genomic-dir", required=True)
parser.add_argument("--cluster", type=int, required=True)
parser.add_argument("--test-year", type=int, default=2008)
parser.add_argument("--budget", type=int, default=300)
parser.add_argument("--max-per-family", type=int, default=5)
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()

print(f"Loading {args.parquet} ...")
df = pd.read_parquet(args.parquet)
marker_cols = gp.marker_columns(df)
df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")

result = fr.build_ranked_recommendations(
    df, marker_cols, args.genomic_dir, cluster=args.cluster, test_year=args.test_year
)
allocation = fr.select_resource_allocation(result, budget=args.budget, max_per_family=args.max_per_family)

print(f"Cluster {args.cluster}: {len(result)} candidate lines across {result['population'].nunique()} families")
print(f"Allocation: {len(allocation)} lines across {allocation['population'].nunique()} families (budget={args.budget}, max_per_family={args.max_per_family})")

os.makedirs(args.out_dir, exist_ok=True)
result.to_csv(f"{args.out_dir}/C{args.cluster}_ranked_recommendations_{args.test_year}.csv", index=False)
allocation.to_csv(f"{args.out_dir}/C{args.cluster}_resource_allocation_{args.test_year}.csv", index=False)
print("Saved.")
