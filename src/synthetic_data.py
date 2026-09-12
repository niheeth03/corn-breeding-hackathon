"""
Generates a small synthetic multi-population, multi-year dataset for judge
mode. The real provided sample (sample_C1_phenotype_100rows.csv etc.) is a
single population in a single year -- fine for testing the data-merge logic,
but the actual modeling here is fundamentally about relationships ACROSS
populations and years (that's the whole finding of this project), which a
one-population sample can't exercise. This generates several small,
genetically-related synthetic populations across a few years so the full
pipeline (training, mate-selection, forward validation, ranking) can run
end-to-end in seconds without touching the real 141MB files.

Not meant to represent real biology precisely -- just enough structure
(shared parents across populations, environment effects, additive marker
effects) to exercise every code path faithfully.
"""

import os
import numpy as np
import pandas as pd

SYNTHETIC_CLUSTER = 9  # fake cluster number so C9.pop.line matches every real ID regex unmodified


def generate_synthetic_dataset(n_populations: int = 12, n_markers: int = 200,
                                progeny_per_pop: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)

    n_founders = max(6, n_populations // 2 + 2)
    founder_ids = [f"SYN_PID{i:04d}" for i in range(n_founders)]
    founder_markers = rng.choice([-1, 0, 1], size=(n_founders, n_markers), p=[0.25, 0.5, 0.25])
    true_effects = rng.normal(0, 1, size=n_markers)

    years = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
    pop_years = [years[i % len(years)] for i in range(n_populations)]

    pheno_rows = []
    genomic_by_pop = {}

    for p in range(1, n_populations + 1):
        f1, f2 = rng.choice(n_founders, size=2, replace=False)
        parent1_markers, parent2_markers = founder_markers[f1], founder_markers[f2]
        year = pop_years[p - 1]

        progeny_markers = np.empty((progeny_per_pop, n_markers), dtype="float32")
        for m in range(n_markers):
            inherit_from_1 = rng.random(progeny_per_pop) < 0.5
            progeny_markers[:, m] = np.where(inherit_from_1, parent1_markers[m], parent2_markers[m])
        flip_mask = rng.random(progeny_markers.shape) < 0.05
        progeny_markers[flip_mask] = rng.choice([-1, 0, 1], size=flip_mask.sum())

        env_effect = rng.normal(0, 5)
        genetic_values = progeny_markers @ true_effects
        noise = rng.normal(0, 8, size=progeny_per_pop)
        yld = 150 + env_effect + genetic_values + noise

        loc = f"SYNLOC{(p % 4) + 1}"
        for i in range(progeny_per_pop):
            pheno_rows.append({
                "YEAR": year, "LOC": loc, "LINE_UNIQUE_ID": f"C{SYNTHETIC_CLUSTER}.{p}.{i+1}",
                "CROSS": f"{founder_ids[f1].replace('SYN_PID','')}/{founder_ids[f2].replace('SYN_PID','')}",
                "YLD_BE": float(yld[i]),
            })

        marker_cols = [f"M{m:05d}" for m in range(n_markers)]
        genomic_df = pd.DataFrame(
            np.vstack([parent1_markers, parent2_markers, progeny_markers]),
            columns=marker_cols,
            index=[founder_ids[f1], founder_ids[f2]] + [f"{i+1:011d}" for i in range(progeny_per_pop)],
        )
        genomic_by_pop[p] = genomic_df

    pheno_df = pd.DataFrame(pheno_rows)
    return pheno_df, genomic_by_pop, [f"M{m:05d}" for m in range(n_markers)]


def write_synthetic_dataset(out_dir: str, **kwargs) -> tuple:
    """Writes a synthetic phenotype CSV and one genomic CSV per population to
    disk, in exactly the same format as the real data, so every existing
    file-based function (parent_marker_matrix, build_ranked_recommendations,
    etc.) works completely unmodified against it."""
    pheno_df, genomic_by_pop, marker_cols = generate_synthetic_dataset(**kwargs)

    genomic_dir = os.path.join(out_dir, f"ImputedPopulationsC{SYNTHETIC_CLUSTER}")
    os.makedirs(genomic_dir, exist_ok=True)
    for pop, gdf in genomic_by_pop.items():
        gdf.to_csv(os.path.join(genomic_dir, f"C{SYNTHETIC_CLUSTER}.{pop}_Imputed.csv"))

    pheno_path = os.path.join(out_dir, f"synthetic_C{SYNTHETIC_CLUSTER}_phenotype.csv")
    pheno_df.to_csv(pheno_path, index=False)

    return pheno_path, genomic_dir, marker_cols
