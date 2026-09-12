"""
Combines Model A (pooled ridge-on-markers, src/genomic_prediction.py) with
Model B (parent-GCA/mid-parent value, src/pedigree_model.py) for the true
cold-start forward prediction: a brand-new family with zero tested progeny.

Design: Model A's between-family signal is unreliable (leave_population_out_cv
showed this directly -- near zero or negative), but its within-family signal
is real (within_population_cv: ~0.6 correlation). Model B is the opposite:
weak overall (~0.10-0.12) but it's the only thing that works at all for
between-family differences with zero progeny data. So:

  combined_prediction(line) = ModelB_family_baseline(its population)
                             + (ModelA_prediction(line) - ModelA_family_average(its population))

Model A supplies ONLY the within-family ranking (de-meaned per family);
Model B supplies ONLY the between-family baseline. Each contributes exactly
the part it's been shown to be good at.
"""

import pandas as pd

from src.genomic_prediction import (
    build_line_dataset, line_marker_matrix, line_level_target,
    fit_gblup, predict_gblup, population_of,
)
from src.pedigree_model import population_level_table, compute_parent_gca, predict_mid_parent_value
from src.validation import compute_metrics, top_k_recovery


def combined_ab_forward_validation(df: pd.DataFrame, marker_cols: list, test_year: int, alpha: float = 1000.0):
    train_year_max = test_year - 1

    # --- Model A: pooled ridge, trained on every year before test_year ---
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=None, year_max=train_year_max)
    model, means = fit_gblup(X_train.values, y_train.values, alpha=alpha)

    X_all = line_marker_matrix(df, marker_cols)
    y_test_year = line_level_target(df, "YLD_BE_ADJ", year_min=test_year, year_max=test_year).dropna()
    common = X_all.index.intersection(y_test_year.index)
    X_test = X_all.loc[common]
    y_test = y_test_year.loc[common]
    modelA_pred = pd.Series(predict_gblup(model, means, X_test.values), index=common)

    pops = pd.Series([population_of(idx) for idx in common], index=common)
    modelA_family_mean = modelA_pred.groupby(pops).transform("mean")
    modelA_within_family_deviation = modelA_pred - modelA_family_mean

    # --- Model B: parent GCA, computed strictly from years before test_year (no leakage) ---
    pretrain_df = df[df["YEAR"] <= train_year_max]
    pop_table = population_level_table(pretrain_df)
    gca = compute_parent_gca(pop_table)
    global_mean = pop_table["own_mean"].mean()

    test_pop_table = population_level_table(df[df["YEAR"] == test_year])
    modelB_by_pop = {}
    for _, row in test_pop_table.iterrows():
        pred, _n_known = predict_mid_parent_value(row["parent1"], row["parent2"], gca, global_mean)
        modelB_by_pop[row["pop"]] = pred
    modelB_pred = pops.map(modelB_by_pop)

    combined_pred = modelB_pred + modelA_within_family_deviation

    def _score(preds, name):
        m = compute_metrics(y_test, preds)
        m["scheme"] = name
        m["top20pct_recovery"] = top_k_recovery(y_test, preds, frac=0.2)
        return m

    return (
        _score(modelA_pred, "modelA_only"),
        _score(modelB_pred, "modelB_only"),
        _score(combined_pred, "modelA_plus_modelB_combined"),
    )
