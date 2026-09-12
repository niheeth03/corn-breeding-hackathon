"""
Proper GBLUP: constructs a genomic relationship matrix (VanRaden's method --
the field-standard way to measure genetic similarity from SNP markers) and
fits it via scikit-learn's KernelRidge, which is mathematically equivalent
to solving the GBLUP mixed model. This replaces treating markers as flat,
independent features (src/genomic_prediction.py's plain ridge) with a model
that reasons in terms of genetic RELATEDNESS -- the thing our own evidence
(within-family works, cross-family identity-based prediction doesn't) says
actually matters here.

Mirrors the same validation schemes as src/validation.py so results are
directly comparable to the plain-ridge numbers we already have.
"""

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

from src.genomic_prediction import population_of
from src.validation import compute_metrics, top_k_recovery


def _center_and_scale(X: np.ndarray):
    """VanRaden's method: center each marker by its own allele-frequency
    based mean, scale by expected heterozygosity so relationships are on a
    comparable scale to a pedigree-based relationship matrix."""
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    allele_freq = (col_mean + 1) / 2  # markers coded -1/0/1
    k = np.sum(2 * allele_freq * (1 - allele_freq))
    if not np.isfinite(k) or k <= 0:
        k = 1.0
    return col_mean, k


def fit_gblup_kernel(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0):
    col_mean, k = _center_and_scale(X_train)
    Z_train = np.where(np.isnan(X_train), col_mean, X_train) - col_mean
    G_train = (Z_train @ Z_train.T) / k
    model = KernelRidge(kernel="precomputed", alpha=alpha)
    model.fit(G_train, y_train)
    aux = {"Z_train": Z_train, "col_mean": col_mean, "k": k}
    return model, aux


def predict_gblup_kernel(model: KernelRidge, aux: dict, X_test: np.ndarray) -> np.ndarray:
    Z_test = np.where(np.isnan(X_test), aux["col_mean"], X_test) - aux["col_mean"]
    G_test_train = (Z_test @ aux["Z_train"].T) / aux["k"]
    return model.predict(G_test_train)


def within_population_cv_gblup(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0,
                                k: int = 5, min_pop_size: int = 20, seed: int = 42) -> dict:
    X_arr = X.values
    y_arr = y.values
    pops = np.array([population_of(idx) for idx in X.index])
    oof = np.full(len(y_arr), np.nan)
    per_pop_corr = {}

    for pop in np.unique(pops):
        pop_idx = np.where(pops == pop)[0]
        if len(pop_idx) < min_pop_size:
            continue
        X_pop, y_pop = X_arr[pop_idx], y_arr[pop_idx]
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        pop_oof = np.full(len(pop_idx), np.nan)
        for train_idx, test_idx in kf.split(X_pop):
            model, aux = fit_gblup_kernel(X_pop[train_idx], y_pop[train_idx], alpha=alpha)
            pop_oof[test_idx] = predict_gblup_kernel(model, aux, X_pop[test_idx])
        oof[pop_idx] = pop_oof
        pm = compute_metrics(y_pop, pop_oof)
        if not np.isnan(pm["correlation"]):
            per_pop_corr[pop] = pm["correlation"]

    valid = ~np.isnan(oof)
    metrics = compute_metrics(y_arr[valid], oof[valid])
    metrics["scheme"] = "within_population_cv_gblup"
    metrics["n_populations"] = len(per_pop_corr)
    metrics["median_per_population_correlation"] = float(np.median(list(per_pop_corr.values()))) if per_pop_corr else np.nan
    metrics["top20pct_recovery"] = top_k_recovery(pd.Series(y_arr[valid]), pd.Series(oof[valid]), frac=0.2)
    return metrics


def leave_population_out_cv_gblup(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0,
                                   max_train_size: int = 6000, max_test_populations: int = 60,
                                   seed: int = 42) -> dict:
    X_arr = X.values
    y_arr = y.values
    pops = np.array([population_of(idx) for idx in X.index])
    rng = np.random.default_rng(seed)

    unique_pops = np.unique(pops)
    if len(unique_pops) > max_test_populations:
        unique_pops = rng.choice(unique_pops, size=max_test_populations, replace=False)

    oof = np.full(len(y_arr), np.nan)
    for pop in unique_pops:
        test_mask = pops == pop
        train_pool = np.where(~test_mask)[0]
        if len(train_pool) < 10 or test_mask.sum() == 0:
            continue
        if len(train_pool) > max_train_size:
            train_pool = rng.choice(train_pool, size=max_train_size, replace=False)
        model, aux = fit_gblup_kernel(X_arr[train_pool], y_arr[train_pool], alpha=alpha)
        oof[test_mask] = predict_gblup_kernel(model, aux, X_arr[test_mask])

    valid = ~np.isnan(oof)
    metrics = compute_metrics(y_arr[valid], oof[valid])
    metrics["scheme"] = "leave_population_out_cv_gblup"
    metrics["n_test_populations"] = len(unique_pops)
    metrics["top20pct_recovery"] = top_k_recovery(pd.Series(y_arr[valid]), pd.Series(oof[valid]), frac=0.2)
    return metrics


def forward_year_validation_gblup(df: pd.DataFrame, marker_cols: list, test_year: int, alpha: float = 1.0):
    from src.genomic_prediction import build_line_dataset, line_marker_matrix, line_level_target

    train_year_max = test_year - 1
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=None, year_max=train_year_max)
    model, aux = fit_gblup_kernel(X_train.values, y_train.values, alpha=alpha)

    X_all = line_marker_matrix(df, marker_cols)
    y_test_year = line_level_target(df, "YLD_BE_ADJ", year_min=test_year, year_max=test_year).dropna()
    common = X_all.index.intersection(y_test_year.index)
    X_test = X_all.loc[common]
    y_test = y_test_year.loc[common]
    preds = pd.Series(predict_gblup_kernel(model, aux, X_test.values), index=common)

    metrics = compute_metrics(y_test, preds)
    metrics["scheme"] = f"forward_{train_year_max}_to_{test_year}_gblup"
    metrics["n_train_lines"] = len(X_train)
    metrics["top20pct_recovery"] = top_k_recovery(y_test, preds, frac=0.2)
    return metrics
