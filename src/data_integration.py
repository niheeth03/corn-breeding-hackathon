"""
Data integration layer for the Corn Breeding hackathon pipeline.

Links three data sources using the keys documented in CORN_BREEDING_DATA_GUIDE.md:
  1. Phenotype <-> Environment : YEAR + LOC
  2. Phenotype <-> Genomic     : LINE_UNIQUE_ID -> population file + line number,
                                 cross-checked against the CROSS column (parent PIDs)
"""

import re
import pandas as pd

# Line numbers are usually "C{cluster}.{population}.{line}", but C2 (and a
# couple of C1 populations) append a trailing replicate/tester-index digit,
# e.g. "C2.1.1.0" or "C1.125.22.2" -- that suffix is not part of the line
# identity, so it's captured but ignored for matching purposes.
LINE_ID_RE = re.compile(r"^C(\d+)\.(\d+)\.(\d+)(?:\.\d+)?$")

# Housekeeping / duplicate columns from the raw phenotype export that carry no
# modeling signal (internal project bookkeeping, duplicate join artifacts).
# Everything else -- every trait, every geographic/env column, every marker --
# is kept, which is the point of "maximizing features."
PHENOTYPE_DROP_COLS = [
    "Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y",
    "projects_x", "projects_y", "ProjectID", "projectID",
    "shorthand_y", "FILE_LIST", "MAB_PROJECT_ID", "YEAR_y", "HG",
]


def load_phenotype(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"LINE": str})
    df = df.rename(columns={"YEAR_x": "YEAR"})
    return df


def load_environmental(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_genomic_population(path: str) -> pd.DataFrame:
    """Load one population's SNP marker file. Index = individual ID (PID... or zero-padded line number).
    A handful of raw files store some progeny IDs unpadded (e.g. 360 instead of
    "00000000360") which pandas then infers as int rather than str -- normalize
    to string so matching logic doesn't depend on formatting consistency."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def parse_line_unique_id(line_unique_id: str):
    """C1.1.191 -> (cluster=1, population=1, line=191)"""
    m = LINE_ID_RE.match(line_unique_id)
    if not m:
        return None
    cluster, population, line = m.groups()
    return int(cluster), int(population), int(line)


def merge_phenotype_environment(pheno: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    return pheno.merge(env, on=["YEAR", "LOC"], how="left", suffixes=("", "_env"))


def split_genomic_parents_progeny(genomic: pd.DataFrame):
    """First two rows of a population genomic file are always the two inbred parents (PID...);
    everything else is zero-padded progeny line numbers ("00000000001", ...)."""
    is_progeny = genomic.index.str.fullmatch(r"\d+")
    parents = genomic.loc[~is_progeny]
    progeny = genomic.loc[is_progeny].copy()
    progeny["line_number"] = progeny.index.astype(int)
    return parents, progeny


def verify_parents_match_cross(parents: pd.DataFrame, cross_value: str) -> bool:
    """Sanity check: the two parent PIDs in the genomic file should match the
    phenotype row's CROSS field (e.g. "1589589/200761"), regardless of order."""
    if not isinstance(cross_value, str) or "/" not in cross_value:
        return False
    expected = set(cross_value.split("/"))
    actual = {pid.replace("PID", "") for pid in parents.index}
    return expected == actual


def merge_phenotype_genomic_for_population(
    pheno_pop: pd.DataFrame, genomic: pd.DataFrame
) -> pd.DataFrame:
    """pheno_pop must already be filtered to a single C{cluster}.{population}."""
    parents, progeny = split_genomic_parents_progeny(genomic)

    pheno_pop = pheno_pop.copy()
    parsed = pheno_pop["LINE_UNIQUE_ID"].apply(parse_line_unique_id)
    pheno_pop["line_number"] = parsed.apply(lambda t: t[2] if t else None)

    merged = pheno_pop.merge(progeny, on="line_number", how="left", suffixes=("", "_marker"))
    return merged, parents


def merge_population_data(cluster: int, population: int, phenotype_data: pd.DataFrame, genomic_dir: str):
    """Full convenience wrapper mirroring the guide's R/Python examples."""
    pattern = f"^C{cluster}\\.{population}\\."
    pheno_subset = phenotype_data[phenotype_data["LINE_UNIQUE_ID"].str.match(pattern)]
    if pheno_subset.empty:
        return pd.DataFrame(), None

    genomic_file = f"{genomic_dir}/C{cluster}.{population}_Imputed.csv"
    genomic = load_genomic_population(genomic_file)
    merged, parents = merge_phenotype_genomic_for_population(pheno_subset, genomic)
    return merged, parents


def prepare_full_feature_table(pheno_env: pd.DataFrame) -> pd.DataFrame:
    """Drop only pure housekeeping/duplicate columns; keep every trait,
    geographic, environmental, and (later) marker column."""
    cols_to_drop = [c for c in PHENOTYPE_DROP_COLS if c in pheno_env.columns]
    return pheno_env.drop(columns=cols_to_drop)


def iter_merged_populations(pheno_env: pd.DataFrame, genomic_dir: str, cluster: int, populations=None):
    """Stream one fully-merged (phenotype + environment + all SNP markers) table
    per population, so the caller never has to hold every population in memory
    at once. `pheno_env` should already be phenotype merged with environment
    (small, cheap) for the whole cluster.

    Yields: (population_id, merged_dataframe, n_markers)
    """
    pheno_env = prepare_full_feature_table(pheno_env)
    parsed = pheno_env["LINE_UNIQUE_ID"].apply(parse_line_unique_id)
    pheno_env = pheno_env.assign(
        _cluster=parsed.apply(lambda t: t[0] if t else None),
        _population=parsed.apply(lambda t: t[1] if t else None),
        line_number=parsed.apply(lambda t: t[2] if t else None),
    )
    pheno_env = pheno_env[pheno_env["_cluster"] == cluster]

    available_pops = sorted(pheno_env["_population"].dropna().unique().astype(int))
    if populations is not None:
        available_pops = [p for p in available_pops if p in set(populations)]

    for pop in available_pops:
        genomic_file = f"{genomic_dir}/C{cluster}.{pop}_Imputed.csv"
        try:
            genomic = load_genomic_population(genomic_file)
        except FileNotFoundError:
            continue

        pheno_pop = pheno_env[pheno_env["_population"] == pop].drop(columns=["_cluster", "_population"])
        _, progeny = split_genomic_parents_progeny(genomic)

        marker_cols = [c for c in genomic.columns]
        progeny[marker_cols] = progeny[marker_cols].astype("float32")

        merged = pheno_pop.merge(progeny, on="line_number", how="left", suffixes=("", "_marker"))
        yield pop, merged, len(marker_cols)
