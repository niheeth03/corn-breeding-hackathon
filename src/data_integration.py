"""
Data integration layer for the Corn Breeding hackathon pipeline.

Links three data sources using the keys documented in CORN_BREEDING_DATA_GUIDE.md:
  1. Phenotype <-> Environment : YEAR + LOC
  2. Phenotype <-> Genomic     : LINE_UNIQUE_ID -> population file + line number,
                                 cross-checked against the CROSS column (parent PIDs)
"""

import re
import pandas as pd

LINE_ID_RE = re.compile(r"^C(\d+)\.(\d+)\.(\d+)$")


def load_phenotype(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"YEAR_x": "YEAR"})
    return df


def load_environmental(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_genomic_population(path: str) -> pd.DataFrame:
    """Load one population's SNP marker file. Index = individual ID (PID... or zero-padded line number)."""
    return pd.read_csv(path, index_col=0)


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
