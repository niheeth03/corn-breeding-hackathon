"""
Model B: cross-family pedigree prediction.

For a population with ZERO tested progeny (the real situation at the start
of a season, before any plots have come back), the only usable signal is
who its two parents are. This estimates each parent's General Combining
Ability (GCA) -- its historical average progeny performance across every
OTHER population it has parented -- and predicts a new population's
expected performance as the mid-parent value (average of its two parents'
GCA). This is classical quantitative-genetics methodology, not something
novel -- it's what breeders did before genomic prediction existed.

Complements Model A (src/genomic_prediction.py's within-population ridge),
which needs at least some of a family's own progeny tested to work at all.
"""

import re
import numpy as np
import pandas as pd

PARENT_RE = re.compile(r"^(\d+)(?:\*\d+)?$")


def parse_parent_ids(cross_str: str):
    """'1589589/200761' -> (1589589, 200761); '200761*2/1376340' -> (200761, 1376340).
    The optional '*N' suffix is a generation/selfing annotation, not a
    different physical parent, so it's stripped before matching."""
    if not isinstance(cross_str, str) or "/" not in cross_str:
        return None, None
    parts = cross_str.split("/")
    if len(parts) != 2:
        return None, None
    ids = []
    for p in parts:
        m = PARENT_RE.match(p.strip())
        ids.append(m.group(1) if m else None)
    return ids[0], ids[1]


def population_level_table(df: pd.DataFrame, adj_col: str = "YLD_BE_ADJ") -> pd.DataFrame:
    """One row per population: its two parent IDs and its mean
    environment-adjusted yield across all its own tested progeny (if any)."""
    d = df.copy()
    d["pop"] = d["LINE_UNIQUE_ID"].str.extract(r"^(C\d+\.\d+)\.")[0]
    parent_ids = d.drop_duplicates("pop")[["pop", "CROSS"]].copy()
    parent_ids[["parent1", "parent2"]] = parent_ids["CROSS"].apply(
        lambda c: pd.Series(parse_parent_ids(c))
    )
    pop_means = d.groupby("pop")[adj_col].mean().rename("own_mean")
    pop_counts = d.groupby("pop")[adj_col].count().rename("own_n")

    table = parent_ids.set_index("pop").join(pop_means).join(pop_counts)
    return table.reset_index()


def compute_parent_gca(pop_table: pd.DataFrame, exclude_pop: str = None) -> pd.Series:
    """Each parent's GCA = average population-mean-yield across every
    population it appears in as parent1 or parent2, optionally excluding one
    population (used during leave-one-population-out validation so a
    population's own data never leaks into its own prediction)."""
    t = pop_table if exclude_pop is None else pop_table[pop_table["pop"] != exclude_pop]
    long = pd.concat([
        t[["parent1", "own_mean"]].rename(columns={"parent1": "parent"}),
        t[["parent2", "own_mean"]].rename(columns={"parent2": "parent"}),
    ])
    long = long.dropna(subset=["parent"])
    return long.groupby("parent")["own_mean"].mean()


def predict_mid_parent_value(parent1: str, parent2: str, gca: pd.Series, global_mean: float) -> tuple:
    """Returns (prediction, n_parents_known) -- n_parents_known lets the
    caller flag how much real signal backs a given prediction (0 = pure
    fallback to the overall mean, 1 = one parent known, 2 = both known)."""
    g1 = gca.get(parent1, np.nan)
    g2 = gca.get(parent2, np.nan)
    known = [g for g in (g1, g2) if not np.isnan(g)]
    if len(known) == 0:
        return global_mean, 0
    return float(np.mean(known)), len(known)


def leave_population_out_pedigree_validation(pop_table: pd.DataFrame) -> dict:
    """Honest validation of Model B: for each population, compute parent GCA
    using every OTHER population, predict this one's mean yield purely from
    its parents, and compare to what it actually did. No progeny data from
    the target population is ever used."""
    global_mean = pop_table["own_mean"].mean()
    preds, actuals, n_known_list = [], [], []

    for _, row in pop_table.iterrows():
        if pd.isna(row["own_mean"]):
            continue
        gca = compute_parent_gca(pop_table, exclude_pop=row["pop"])
        pred, n_known = predict_mid_parent_value(row["parent1"], row["parent2"], gca, global_mean)
        preds.append(pred)
        actuals.append(row["own_mean"])
        n_known_list.append(n_known)

    preds = np.array(preds)
    actuals = np.array(actuals)
    n_known_arr = np.array(n_known_list)

    def _corr_rmse(mask):
        if mask.sum() < 2:
            return np.nan, np.nan
        c = np.corrcoef(actuals[mask], preds[mask])[0, 1]
        r = float(np.sqrt(np.mean((actuals[mask] - preds[mask]) ** 2)))
        return float(c), r

    overall_corr, overall_rmse = _corr_rmse(np.ones(len(preds), dtype=bool))
    both_corr, both_rmse = _corr_rmse(n_known_arr == 2)
    one_corr, one_rmse = _corr_rmse(n_known_arr == 1)

    return {
        "n_populations": len(preds),
        "overall_correlation": overall_corr,
        "overall_rmse": overall_rmse,
        "n_both_parents_known": int((n_known_arr == 2).sum()),
        "both_parents_known_correlation": both_corr,
        "both_parents_known_rmse": both_rmse,
        "n_one_parent_known": int((n_known_arr == 1).sum()),
        "one_parent_known_correlation": one_corr,
        "one_parent_known_rmse": one_rmse,
        "n_no_parents_known": int((n_known_arr == 0).sum()),
    }
