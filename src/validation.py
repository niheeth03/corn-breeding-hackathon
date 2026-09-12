"""
Validation schemes for the genomic prediction model, and the headline
comparison of the whole project: a naive random CV split silently leaks
family/sibling information (inflating accuracy), while a leave-population-out
split is the honest test of "can this predict a brand-new family." Both are
computed the same way so the gap between them is directly comparable.

Also implements the real forward-in-time validation: train only on
2001-2007 data, predict 2008 line values, and score against the 2008 yields
that are already sitting in the raw dataset.

Everything below converts to plain numpy arrays ONCE up front and then
indexes with integer positions inside the CV loops -- fitting/predicting
thousands of times against a ~2900-column pandas DataFrame (with fillna/
reindex on every call) is dramatically slower than the equivalent numpy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.genomic_prediction import fit_gblup, predict_gblup, population_of


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"n": len(y_true), "correlation": np.nan, "rmse": np.nan}
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"n": len(y_true), "correlation": float(corr), "rmse": rmse}


def naive_random_cv(X: pd.DataFrame, y: pd.Series, k: int = 5, alpha: float = 100.0, seed: int = 42) -> dict:
    """Random K-fold across individual LINES, ignoring which population/
    family they belong to. Siblings from the same biparental cross can end
    up split across train and test, so the model can partly learn
    family-level shortcuts rather than a true marker-effect model --
    this is expected to look optimistic."""
    X_arr = X.values
    y_arr = y.values
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y_arr), np.nan)
    for train_idx, test_idx in kf.split(X_arr):
        model, means = fit_gblup(X_arr[train_idx], y_arr[train_idx], alpha=alpha)
        oof[test_idx] = predict_gblup(model, means, X_arr[test_idx])
    oof_preds = pd.Series(oof, index=y.index)
    metrics = compute_metrics(y_arr, oof)
    metrics["scheme"] = "naive_random_cv"
    return metrics, oof_preds


def leave_population_out_cv(X: pd.DataFrame, y: pd.Series, alpha: float = 100.0,
                             max_train_size: int = 6000, max_test_populations: int = 60,
                             seed: int = 42) -> dict:
    """Train on every OTHER population's lines, predict this one -- the
    model has never seen this family's genetic background at all, which is
    the real situation for a brand-new 2008 population.

    Refitting on ~all ~40k lines (n > p, so cost scales with n*p^2) once per
    held-out population is computationally wasteful for what's a diagnostic/
    supporting scheme (not the deployed model): we cap the training pool to
    a random subsample and test on a random subset of populations rather
    than all of them. This doesn't change the qualitative finding (does
    cross-family signal exist at all) and keeps runtime bounded regardless
    of dataset size."""
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
        model, means = fit_gblup(X_arr[train_pool], y_arr[train_pool], alpha=alpha)
        oof[test_mask] = predict_gblup(model, means, X_arr[test_mask])

    oof_preds = pd.Series(oof, index=y.index)
    valid = ~np.isnan(oof)
    metrics = compute_metrics(y_arr[valid], oof[valid])
    metrics["scheme"] = "leave_population_out_cv"
    metrics["n_test_populations"] = len(unique_pops)
    return metrics, oof_preds


def within_population_cv(X: pd.DataFrame, y: pd.Series, alpha: float = 100.0,
                          k: int = 5, min_pop_size: int = 20, seed: int = 42) -> dict:
    """The realistic deployable scenario: within EACH family, train on a
    subset of its own tested progeny and predict the rest of that SAME
    family. Biparental LD is family-specific and high, so this is where
    genomic prediction is expected to actually work -- unlike pooling raw
    marker values across genetically unrelated families (leave_population_out_cv),
    which mixes in family-identity effects rather than true marker effects.
    Directly serves the resource-allocation story: phenotype fewer
    plots per family, use markers to rank the untested remainder."""
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
            model, means = fit_gblup(X_pop[train_idx], y_pop[train_idx], alpha=alpha)
            pop_oof[test_idx] = predict_gblup(model, means, X_pop[test_idx])
        oof[pop_idx] = pop_oof
        pop_metrics = compute_metrics(y_pop, pop_oof)
        if not np.isnan(pop_metrics["correlation"]):
            per_pop_corr[pop] = pop_metrics["correlation"]

    oof_preds = pd.Series(oof, index=y.index)
    valid = ~np.isnan(oof)
    metrics = compute_metrics(y_arr[valid], oof[valid])
    metrics["scheme"] = "within_population_cv"
    metrics["n_populations"] = len(per_pop_corr)
    metrics["median_per_population_correlation"] = float(np.median(list(per_pop_corr.values()))) if per_pop_corr else np.nan
    return metrics, oof_preds, per_pop_corr


def tune_alpha_within_population(X: pd.DataFrame, y: pd.Series, alphas=None,
                                  k: int = 5, min_pop_size: int = 20) -> tuple:
    """Grid search alpha using within_population_cv as the target metric --
    that's the scheme where signal genuinely exists, so it's the
    informative one to tune against (leave_population_out_cv stays near
    zero regardless of alpha, since the problem there is lack of shared
    genetic signal, not regularization strength)."""
    if alphas is None:
        alphas = [1, 10, 50, 100, 300, 1000, 3000, 10000]
    results = []
    for alpha in alphas:
        metrics, _, _ = within_population_cv(X, y, alpha=alpha, k=k, min_pop_size=min_pop_size)
        results.append((alpha, metrics["correlation"]))
        print(f"    alpha={alpha:>7}: within-population correlation = {metrics['correlation']:.4f}", flush=True)
    best_alpha = max(results, key=lambda t: (t[1] if not np.isnan(t[1]) else -np.inf))[0]
    return best_alpha, results


def top_k_recovery(y_true, y_pred, frac: float = 0.2) -> dict:
    """Of the actual best `frac` of lines, what fraction does the model's
    predicted top `frac` actually catch? More decision-relevant than
    correlation for a selection problem -- a breeder cares about recovering
    the best lines, not the correlation coefficient, and this metric is
    immune to the calibration issues correlation/RMSE can disagree on
    (only ranking matters, not predicted magnitude)."""
    y_true = pd.Series(y_true).dropna()
    y_pred = pd.Series(y_pred).reindex(y_true.index).dropna()
    common = y_true.index.intersection(y_pred.index)
    y_true, y_pred = y_true.loc[common], y_pred.loc[common]

    n = len(y_true)
    k = max(1, int(np.ceil(n * frac)))
    if n < 2:
        return {"n": n, "k": k, "recovery": np.nan, "chance_baseline": frac}

    true_top = set(y_true.sort_values(ascending=False).index[:k])
    pred_top = set(y_pred.sort_values(ascending=False).index[:k])
    recovery = len(true_top & pred_top) / k
    return {"n": n, "k": k, "recovery": recovery, "chance_baseline": frac}


def forward_year_validation(df: pd.DataFrame, marker_cols: list, test_year: int, alpha: float = 100.0):
    """General year-forward test: train on every year BEFORE test_year,
    predict test_year's lines from DNA alone, score against what actually
    happened. Used as both an internal rehearsal (test_year=2006, 2007 --
    years we already fully know, used purely to build confidence in the
    method) and the real final check (test_year=2008)."""
    from src.genomic_prediction import build_line_dataset, line_marker_matrix, line_level_target

    train_year_max = test_year - 1
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=None, year_max=train_year_max)
    model, means = fit_gblup(X_train.values, y_train.values, alpha=alpha)

    X_all = line_marker_matrix(df, marker_cols)
    y_test_year = line_level_target(df, "YLD_BE_ADJ", year_min=test_year, year_max=test_year).dropna()

    common = X_all.index.intersection(y_test_year.index)
    X_test = X_all.loc[common]
    y_test = y_test_year.loc[common]
    preds = pd.Series(predict_gblup(model, means, X_test.values), index=common)

    metrics = compute_metrics(y_test, preds)
    metrics["scheme"] = f"forward_{train_year_max}_to_{test_year}"
    metrics["train_year_max"] = train_year_max
    metrics["test_year"] = test_year
    metrics["n_train_lines"] = len(X_train)
    metrics["top20pct_recovery"] = top_k_recovery(y_test, preds, frac=0.2)

    # null baseline: same forward split, but just predict the training mean
    null_pred = pd.Series(y_train.mean(), index=common)
    null_metrics = compute_metrics(y_test, null_pred)
    null_metrics["scheme"] = f"forward_{train_year_max}_to_{test_year}_null_baseline"

    # phenotypic baseline: for lines also observed before test_year, use
    # their own historical mean adjusted yield instead of markers
    hist_mean = df[df["YEAR"] <= train_year_max].groupby("LINE_UNIQUE_ID")["YLD_BE_ADJ"].mean()
    pheno_pred = y_test.index.to_series().map(hist_mean)
    pheno_metrics = compute_metrics(y_test, pheno_pred)
    pheno_metrics["scheme"] = f"forward_{train_year_max}_to_{test_year}_phenotypic_baseline"
    pheno_metrics["n_lines_with_history"] = pheno_pred.notna().sum()

    return metrics, null_metrics, pheno_metrics, preds, y_test


def forward_2008_validation(df: pd.DataFrame, marker_cols: list, alpha: float = 100.0):
    """Back-compat wrapper -- the original 2007->2008 check."""
    return forward_year_validation(df, marker_cols, test_year=2008, alpha=alpha)


def forward_raw_yield_vs_farm_mean(df: pd.DataFrame, marker_cols: list, alpha: float = 100.0):
    """The exact comparison behind 'SNPs did not beat farm means': does
    adding the genomic model's predicted deviation on top of a pure
    farm-mean (YEAR, LOC) baseline reduce error on RAW yield, versus the
    farm-mean baseline alone? This has to be done on raw yield, not our
    environment-adjusted target, since the adjustment already removes the
    farm-mean's explanatory power before any modeling -- comparing on the
    adjusted scale would be apples-to-oranges.

    Train window: 2001-2007. Test: 2008 raw YLD_BE, aggregated to one
    prediction and one actual value per line (averaged across whichever
    2008 environments that line was actually tested in)."""
    from src.genomic_prediction import build_line_dataset, line_marker_matrix
    from src.baselines import environmental_mean_baseline

    train_df = df[(df["YEAR"] >= 2001) & (df["YEAR"] <= 2007)]
    test_df = df[df["YEAR"] == 2008].copy()

    # farm-mean prediction per OBSERVATION, then averaged up to per-LINE
    test_df["farm_mean_pred"] = environmental_mean_baseline(train_df, test_df, trait="YLD_BE").values
    farm_mean_per_line = test_df.groupby("LINE_UNIQUE_ID")["farm_mean_pred"].mean()
    actual_raw_per_line = test_df.groupby("LINE_UNIQUE_ID")["YLD_BE"].mean().dropna()

    # genomic model's predicted deviation, trained on the same window as the main forward test
    X_train, y_train = build_line_dataset(df, marker_cols, "YLD_BE_ADJ", year_min=2001, year_max=2007)
    model, means = fit_gblup(X_train.values, y_train.values, alpha=alpha)
    X_all = line_marker_matrix(df, marker_cols)

    common = (X_all.index
              .intersection(farm_mean_per_line.index)
              .intersection(actual_raw_per_line.index))
    genomic_deviation = pd.Series(predict_gblup(model, means, X_all.loc[common].values), index=common)

    farm_only_pred = farm_mean_per_line.loc[common]
    combined_pred = farm_only_pred + genomic_deviation
    actual = actual_raw_per_line.loc[common]

    farm_only_metrics = compute_metrics(actual, farm_only_pred)
    farm_only_metrics["scheme"] = "farm_mean_only"
    farm_only_metrics["top20pct_recovery"] = top_k_recovery(actual, farm_only_pred, frac=0.2)

    combined_metrics = compute_metrics(actual, combined_pred)
    combined_metrics["scheme"] = "farm_mean_plus_genomic_deviation"
    combined_metrics["top20pct_recovery"] = top_k_recovery(actual, combined_pred, frac=0.2)

    return farm_only_metrics, combined_metrics
