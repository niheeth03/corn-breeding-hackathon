"""
Genomic prediction core: ridge regression on SNP markers to estimate a
per-line breeding value (GEBV) for yield, after removing environmental
(year x location) effects -- the standard two-stage approximation to GBLUP
that's tractable without a full mixed-model solver.

Stage A: adjust each observation for its environment's mean yield.
Stage B: aggregate to one env-adjusted yield per line, and fit
         Ridge(markers) -> adjusted yield. Ridge regression on centered
         markers is mathematically equivalent to RR-BLUP up to a scaling of
         the penalty, which is why it's the standard fast substitute for a
         full GBLUP mixed model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def marker_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("M0")]


def add_environment_adjusted_yield(df: pd.DataFrame, trait: str = "YLD_BE") -> pd.DataFrame:
    """Subtract each row's YEAR x LOC trial mean from its trait value, using
    ALL phenotype records in that environment (not just genotyped ones) so
    the environment mean is estimated as precisely as possible."""
    df = df.copy()
    env_mean = df.groupby(["YEAR", "LOC"])[trait].transform("mean")
    df[f"{trait}_ADJ"] = df[trait] - env_mean
    return df


def line_marker_matrix(df: pd.DataFrame, marker_cols: list) -> pd.DataFrame:
    """One row per LINE_UNIQUE_ID -- markers don't vary by environment, so
    we just need each genotyped line's marker vector once."""
    has_markers = df[marker_cols[0]].notna()
    m = (
        df.loc[has_markers, ["LINE_UNIQUE_ID"] + marker_cols]
        .drop_duplicates("LINE_UNIQUE_ID")
        .set_index("LINE_UNIQUE_ID")
    )
    return m


def line_level_target(
    df: pd.DataFrame, target_col: str, year_min: int = None, year_max: int = None
) -> pd.Series:
    """Mean environment-adjusted yield per line, optionally restricted to a
    year window (used to build a strict pre-2008 training target vs. a
    2008-only evaluation target from the SAME underlying data)."""
    d = df
    if year_min is not None:
        d = d[d["YEAR"] >= year_min]
    if year_max is not None:
        d = d[d["YEAR"] <= year_max]
    return d.groupby("LINE_UNIQUE_ID")[target_col].mean()


def build_line_dataset(df: pd.DataFrame, marker_cols: list, target_col: str,
                        year_min: int = None, year_max: int = None):
    """Return (X, y) aligned on LINE_UNIQUE_ID for lines that have both
    genomic markers and at least one phenotype record in the year window."""
    X = line_marker_matrix(df, marker_cols)
    y = line_level_target(df, target_col, year_min, year_max).dropna()
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


def fit_gblup(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 100.0):
    """Fit ridge regression on markers (plain numpy arrays -- this is called
    thousands of times across CV folds/populations/alphas, and pandas
    fillna/reindex on a ~2900-column frame is far too slow to do that often).
    Missing genotype calls are imputed with the training-set marker mean
    (the standard genomic-prediction convention -- imputing towards 0 net
    additive effect at that locus), using ONLY training data to avoid
    leakage into validation folds."""
    col_mean = np.nanmean(X_train, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)  # marker missing for ALL training rows falls back to 0
    X_filled = np.where(np.isnan(X_train), col_mean, X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_filled, y_train)
    return model, col_mean


def predict_gblup(model: Ridge, col_mean: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_filled = np.where(np.isnan(X), col_mean, X)
    return model.predict(X_filled)


def population_of(line_unique_id: str):
    parts = line_unique_id.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else None
