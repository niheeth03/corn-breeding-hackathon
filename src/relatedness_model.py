"""
Similarity-weighted prediction: predict a population's performance as a
weighted average of every OTHER known population's actual performance,
weighted by genetic similarity between their parents' own DNA -- not a
global linear model (which we proved is equivalent to plain ridge
regardless of primal/dual form), but a genuinely different mechanism:
closely related families count heavily, unrelated ones barely count at all.

Scoped at the POPULATION level (not individual-level) deliberately --
we already hit real compute blowups doing individual-level relationship
matrices at ~40k rows; at most a few hundred populations is cheap.
"""

import numpy as np
import pandas as pd

from src.genomic_prediction import population_of


def population_fingerprint(genomic_dir: str, cluster: int, population: int) -> np.ndarray:
    """Average marker vector of a population's two founder parents. Reads
    only the first two rows of the raw genomic file (the parent rows) --
    fast, doesn't touch the hundreds of progeny rows below them."""
    path = f"{genomic_dir}/C{cluster}.{population}_Imputed.csv"
    df = pd.read_csv(path, index_col=0, nrows=2)
    return df.mean(axis=0).values.astype("float32")


def build_fingerprint_table(genomic_dir: str, cluster: int, populations) -> pd.DataFrame:
    """One row per population (index = population number as int), columns = markers."""
    rows = {}
    for pop in populations:
        try:
            rows[pop] = population_fingerprint(genomic_dir, cluster, pop)
        except FileNotFoundError:
            continue
    return pd.DataFrame(rows).T


def similarity_matrix(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Population x population genetic similarity: correlation between
    parent-average marker profiles (pairwise-complete, so a marker missing
    for one population's parents doesn't drop the whole comparison)."""
    return fingerprints.T.corr()


def population_year_target_table(df: pd.DataFrame, adj_col: str = "YLD_BE_ADJ") -> pd.DataFrame:
    """One row per population: its cluster-relative population number, its
    YEAR (every population belongs to exactly one), and its mean
    environment-adjusted yield across its own tested progeny."""
    d = df.copy()
    d["pop_full"] = d["LINE_UNIQUE_ID"].apply(population_of)
    d["pop_num"] = d["pop_full"].str.extract(r"\.(\d+)$")[0].astype(int)
    grouped = d.groupby("pop_num").agg(YEAR=("YEAR", "first"), own_mean=(adj_col, "mean"), own_n=(adj_col, "count"))
    return grouped.reset_index()


def kernel_weighted_predict(target_pop: int, known_pops: list, sim: pd.DataFrame,
                             known_means: pd.Series, bandwidth: float) -> float:
    """Weighted average of known populations' own_mean, weighted by
    exp((similarity - 1) / bandwidth) -- identical parents (similarity=1)
    get full weight, weight decays smoothly as similarity drops. Smaller
    bandwidth = sharper (only close relatives matter); larger = smoother
    (closer to a flat average of everyone)."""
    valid = [p for p in known_pops if p in sim.columns]
    if not valid:
        return known_means.mean()
    sims = sim.loc[target_pop, valid].values
    means = known_means.loc[valid].values
    weights = np.exp((sims - 1.0) / bandwidth)
    weights = np.nan_to_num(weights, nan=0.0)
    if weights.sum() <= 1e-12:
        return known_means.mean()
    return float(np.sum(weights * means) / np.sum(weights))


def year_forward_relatedness_validation(pop_table: pd.DataFrame, sim: pd.DataFrame,
                                         test_year: int, bandwidth: float) -> dict:
    """Train (known) = every population from a year strictly before
    test_year. Predict every test_year population purely from genetic
    similarity to those known populations. No same-year or future data
    ever touches the prediction."""
    known = pop_table[pop_table["YEAR"] < test_year].dropna(subset=["own_mean"])
    known_means = known.set_index("pop_num")["own_mean"]
    known_pops = known_means.index.tolist()

    target = pop_table[pop_table["YEAR"] == test_year].dropna(subset=["own_mean"])

    preds, actuals = [], []
    for _, row in target.iterrows():
        pop = row["pop_num"]
        if pop not in sim.index:
            continue
        pred = kernel_weighted_predict(pop, known_pops, sim, known_means, bandwidth)
        preds.append(pred)
        actuals.append(row["own_mean"])

    preds = np.array(preds)
    actuals = np.array(actuals)
    if len(preds) < 2:
        return {"n": len(preds), "correlation": np.nan, "rmse": np.nan, "bandwidth": bandwidth, "test_year": test_year}

    corr = float(np.corrcoef(actuals, preds)[0, 1])
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

    from src.validation import top_k_recovery
    recovery = top_k_recovery(pd.Series(actuals), pd.Series(preds), frac=0.2)

    return {"n": len(preds), "correlation": corr, "rmse": rmse, "bandwidth": bandwidth,
            "test_year": test_year, "top20pct_recovery": recovery}
