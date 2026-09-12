"""
Genomic mate/cross prediction -- an established technique from the genomic
selection literature, purpose-built for exactly our situation: predicting a
brand-new cross's value using ONLY its two parents' own DNA, before a
single progeny has ever been tested.

Mechanism: train the marker-effect model (same ridge/GBLUP model as
src/genomic_prediction.py) on progeny from OTHER, already-tested families.
Then apply that TRAINED model directly to the two PARENT lines' own
genotypes (which sit in every population's genomic file regardless of
whether that population has been phenotyped) to get each parent's own
estimated breeding value (GEBV). Predict the new cross's expected value as
the average of its two parents' GEBVs -- the "mid-parent" prediction,
classical quantitative genetics, now computed from real marker effects
instead of raw historical averaging (which is what src/pedigree_model.py
did).

This directly matches this dataset's own stated purpose: "GCA assessment."
"""

import numpy as np
import pandas as pd

from src.genomic_prediction import fit_gblup, predict_gblup, build_line_dataset


def parent_marker_matrix(genomic_dir: str, cluster: int, population: int) -> np.ndarray:
    """The two parent rows of a population's raw genomic file -- always
    available, since genotyping doesn't require phenotyping."""
    path = f"{genomic_dir}/C{cluster}.{population}_Imputed.csv"
    df = pd.read_csv(path, index_col=0, nrows=2)
    return df.values.astype("float32")


def mid_parent_gebv_prediction(genomic_dir: str, cluster: int, population: int,
                                model, means) -> float:
    parent_markers = parent_marker_matrix(genomic_dir, cluster, population)
    gebvs = predict_gblup(model, means, parent_markers)
    return float(np.mean(gebvs))


def mate_selection_forward_validation(df: pd.DataFrame, marker_cols: list, genomic_dir: str,
                                       cluster: int, test_year: int, alpha: float = 1000.0):
    """Train Model A's marker-effect model on progeny from years before
    test_year, then predict test_year's populations using ONLY their
    parents' own DNA -- never any of that population's own progeny data."""
    from src.relatedness_model import population_year_target_table
    from src.validation import compute_metrics, top_k_recovery

    train_year_max = test_year - 1
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=None, year_max=train_year_max)
    model, means = fit_gblup(X_train.values, y_train.values, alpha=alpha)

    pop_table = population_year_target_table(df)
    target = pop_table[pop_table["YEAR"] == test_year].dropna(subset=["own_mean"])

    preds, actuals = [], []
    for _, row in target.iterrows():
        pop = int(row["pop_num"])
        try:
            pred = mid_parent_gebv_prediction(genomic_dir, cluster, pop, model, means)
        except FileNotFoundError:
            continue
        preds.append(pred)
        actuals.append(row["own_mean"])

    preds = pd.Series(preds)
    actuals = pd.Series(actuals)
    metrics = compute_metrics(actuals, preds)
    metrics["scheme"] = f"mate_selection_{train_year_max}_to_{test_year}"
    metrics["n_train_lines"] = len(X_train)
    metrics["top20pct_recovery"] = top_k_recovery(actuals, preds, frac=0.2)
    return metrics
