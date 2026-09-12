"""
The actual deliverable: a ranked list of specific 2008 lines to advance.

Combines the two things we've validated actually work:
  - mate_selection_model: best-validated FAMILY-level baseline (mid-parent
    GEBV), works even for a family with zero tested progeny.
  - genomic_prediction (Model A): real within-family ranking signal, once
    centered relative to its own family average to strip out its unreliable
    between-family component (which mate-selection already covers better).

final_prediction(line) = mate_selection_family_baseline(its population)
                        + (ModelA_prediction(line) - ModelA_family_average(its population))

Works for every genotyped 2008 line regardless of whether it has been
phenotyped yet -- genotyping doesn't require phenotyping, which is exactly
why this is usable on the real January-2008 decision date.
"""

import numpy as np
import pandas as pd

from src.genomic_prediction import (
    build_line_dataset, fit_gblup, predict_gblup, population_of,
)
from src.mate_selection_model import parent_marker_matrix
from src.data_integration import load_genomic_population, split_genomic_parents_progeny

# RMSE of the exact final hybrid formula, measured end-to-end against real
# 2008 outcomes (n=7429 for C1, n=8533 for C2 -- nearly every genotyped
# candidate line). Used to build an approximate 68%/95% prediction interval
# around each line's point prediction.
VALIDATED_RMSE = {1: 10.920, 2: 11.019}


def build_ranked_recommendations(df: pd.DataFrame, marker_cols: list, genomic_dir: str,
                                  cluster: int, test_year: int = 2008, alpha: float = 1000.0) -> pd.DataFrame:
    train_year_max = test_year - 1

    # Train Model A on every year before test_year (the real, final training set) --
    # training correctly stays phenotype-anchored, since it needs real yield to learn from.
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=None, year_max=train_year_max)
    model, means = fit_gblup(X_train.values, y_train.values, alpha=alpha)

    # Which populations belong to test_year -- this mapping itself is necessarily
    # phenotype-anchored (year is a phenotype-trial concept), but once we know
    # the population, we predict for EVERY genotyped progeny in its raw genomic
    # file, not just the subset that happens to already have a phenotype row.
    test_year_pop_nums = sorted(set(
        int(p) for p in
        df.loc[df["YEAR"] == test_year, "LINE_UNIQUE_ID"].apply(population_of).str.extract(r"\.(\d+)$")[0].dropna()
    ))

    rows = []
    family_baseline = {}
    modelA_preds_by_pop = {}
    for pop_num in test_year_pop_nums:
        try:
            genomic = load_genomic_population(f"{genomic_dir}/C{cluster}.{pop_num}_Imputed.csv")
        except FileNotFoundError:
            continue
        parents, progeny = split_genomic_parents_progeny(genomic)
        if len(progeny) == 0:
            continue

        marker_matrix = progeny[marker_cols].astype("float32").values if set(marker_cols).issubset(progeny.columns) else progeny.values.astype("float32")
        modelA_pred = predict_gblup(model, means, marker_matrix)
        modelA_preds_by_pop[pop_num] = modelA_pred

        gebvs = predict_gblup(model, means, parents[marker_cols].astype("float32").values if set(marker_cols).issubset(parents.columns) else parents.values.astype("float32"))
        family_baseline[pop_num] = float(np.mean(gebvs))

        family_avg_pred = float(np.mean(modelA_pred))
        line_ids = [f"C{cluster}.{pop_num}.{int(idx)}" for idx in progeny.index]
        for line_id, pred in zip(line_ids, modelA_pred):
            rows.append({
                "LINE_UNIQUE_ID": line_id,
                "population": f"C{cluster}.{pop_num}",
                "family_baseline_mate_selection": family_baseline[pop_num],
                "modelA_individual_prediction": pred,
                "within_family_deviation": pred - family_avg_pred,
                "final_predicted_yield_advantage": family_baseline[pop_num] + (pred - family_avg_pred),
            })

    if not rows:
        raise ValueError(f"No genotyped lines found for test_year={test_year}")

    result = pd.DataFrame(rows).dropna(subset=["final_predicted_yield_advantage"])

    # Uncertainty band from the validated end-to-end RMSE of this exact formula
    rmse = VALIDATED_RMSE.get(cluster, float(np.sqrt(np.mean((y_train.values - predict_gblup(model, means, X_train.values)) ** 2))))
    result["predicted_lower_68pct"] = result["final_predicted_yield_advantage"] - rmse
    result["predicted_upper_68pct"] = result["final_predicted_yield_advantage"] + rmse
    result["predicted_lower_95pct"] = result["final_predicted_yield_advantage"] - 1.96 * rmse
    result["predicted_upper_95pct"] = result["final_predicted_yield_advantage"] + 1.96 * rmse

    # Confidence tier: how much real parent history backs this family's baseline
    from src.pedigree_model import population_level_table
    train_df = df[df["YEAR"] <= train_year_max]
    pop_table = population_level_table(train_df)
    known_parents = set(pop_table["parent1"].dropna()) | set(pop_table["parent2"].dropna())

    test_pop_table = population_level_table(df[df["YEAR"] == test_year])
    tier_by_pop = {}
    for _, row in test_pop_table.iterrows():
        n_known = int(row["parent1"] in known_parents) + int(row["parent2"] in known_parents)
        tier_by_pop[int(row["pop"].split(".")[-1])] = {0: "Low", 1: "Medium", 2: "High"}[n_known]
    result["confidence_tier"] = result["population"].apply(lambda p: tier_by_pop.get(int(p.split(".")[-1]), "Low"))

    result = result.sort_values("final_predicted_yield_advantage", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    result["percentile"] = 100 * (1 - result.index / len(result))
    return result


def select_resource_allocation(ranked_df: pd.DataFrame, budget: int, max_per_family: int = 5) -> pd.DataFrame:
    """Pure rank-order selection puts everything into whichever single
    family happens to score highest -- real breeding programs protect
    genetic diversity rather than betting the whole budget on one cross.
    Caps how many plots any one family can claim, then fills remaining
    budget by rank among what's left."""
    selected = []
    family_counts = {}
    for _, row in ranked_df.iterrows():
        if len(selected) >= budget:
            break
        fam = row["population"]
        if family_counts.get(fam, 0) >= max_per_family:
            continue
        selected.append(row)
        family_counts[fam] = family_counts.get(fam, 0) + 1
    return pd.DataFrame(selected).reset_index(drop=True)
