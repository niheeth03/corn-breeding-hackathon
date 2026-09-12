"""
JUDGE MODE -- end-to-end pipeline demo, runs in seconds, no full 141MB
dataset required.

What this does:
  1. Generates a small synthetic multi-population, multi-year dataset (the
     real single-population sample files can't exercise cross-family /
     cross-year behavior, which is the actual subject of this project).
  2. Runs the SAME merge code used on the real data (src/data_integration.py)
     to build phenotype+genomic training data.
  3. Trains the genomic model, builds the mate-selection (parent-GEBV)
     family baseline, and combines them into the final line-level ranking
     -- the exact same code path used to produce the real results.
  4. Prints validation metrics and a resource-allocation recommendation.

The REAL results (built the same way, on the actual ~500-population-per-
cluster dataset via HPRC) are already saved in outputs/C1_ranked_
recommendations_2008.csv and outputs/C2_ranked_recommendations_2008.csv --
this script demonstrates the method works, it does not regenerate those.

Usage:
    python run_pipeline.py
"""

import os
import shutil
import tempfile

import pandas as pd

from src import genomic_prediction as gp
from src import data_integration as di
from src import validation as val
from src import final_recommendation as fr
from src import synthetic_data as synth


def main():
    print("=" * 70)
    print("JUDGE MODE: end-to-end pipeline on synthetic data")
    print("=" * 70)

    scratch_dir = tempfile.mkdtemp(prefix="corn_judge_mode_")
    try:
        print(f"\n[1/5] Generating synthetic multi-population dataset in {scratch_dir} ...")
        pheno_path, genomic_dir, marker_cols = synth.write_synthetic_dataset(
            scratch_dir, n_populations=16, n_markers=200, progeny_per_pop=40, seed=0
        )
        pheno = pd.read_csv(pheno_path)
        print(f"      {len(pheno)} phenotype rows across {pheno['LINE_UNIQUE_ID'].nunique()} lines, "
              f"years {pheno['YEAR'].min()}-{pheno['YEAR'].max()}")

        print("\n[2/5] Merging phenotype + genomic data (same code as the real pipeline) ...")
        merged_parts = []
        for pop, merged, n_markers in di.iter_merged_populations(pheno, genomic_dir, synth.SYNTHETIC_CLUSTER):
            merged_parts.append(merged)
        df = pd.concat(merged_parts, ignore_index=True)
        print(f"      Merged table: {len(df)} rows x {len(df.columns)} columns ({n_markers} markers)")

        print("\n[3/5] Environment-adjusting yield and running validation schemes ...")
        df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")
        X, y = gp.build_line_dataset(df, marker_cols, "YLD_BE_ADJ")
        naive, _ = val.naive_random_cv(X, y, alpha=1000.0, k=3)
        wpop, _, _ = val.within_population_cv(X, y, alpha=1000.0, k=3, min_pop_size=10)
        print(f"      Naive random CV (leaky, reference only): correlation={naive['correlation']:.3f}")
        print(f"      Within-population CV (real signal):      correlation={wpop['correlation']:.3f}")

        print("\n[4/5] Building final ranked recommendations for the last synthetic year ...")
        test_year = int(pheno["YEAR"].max())
        result = fr.build_ranked_recommendations(
            df, marker_cols, genomic_dir, cluster=synth.SYNTHETIC_CLUSTER, test_year=test_year, alpha=1000.0
        )
        print(f"      {len(result)} candidate lines ranked across {result['population'].nunique()} families")

        print("\n[5/5] Applying a resource-allocation budget (10 plots, max 2/family) ...")
        allocation = fr.select_resource_allocation(result, budget=10, max_per_family=2)
        print(allocation[["LINE_UNIQUE_ID", "population", "final_predicted_yield_advantage",
                           "confidence_tier"]].to_string(index=False))

        out_path = "outputs/judge_mode_demo_recommendations.csv"
        os.makedirs("outputs", exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"\nFull demo ranking saved to {out_path}")

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("This demo used small synthetic data purely to exercise every step")
    print("of the pipeline quickly. The REAL results, built the same way on")
    print("the actual dataset (~500 populations/cluster, 2001-2008) via HPRC,")
    print("are already saved at:")
    print("  outputs/C1_ranked_recommendations_2008.csv")
    print("  outputs/C1_resource_allocation_2008.csv")
    print("  outputs/C2_ranked_recommendations_2008.csv")
    print("  outputs/C2_resource_allocation_2008.csv")
    print("See README.md for the full validated results and methodology.")
    print("=" * 70)


if __name__ == "__main__":
    main()
