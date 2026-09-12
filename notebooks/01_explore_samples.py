"""Phase 1 sanity check: confirm phenotype<->environment and phenotype<->genomic
joins actually work on the provided sample files before building anything on top."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data_integration as di

PHENO_PATH = "sample_data/sample_C1_phenotype_100rows.csv"
ENV_PATH = "sample_data/environmental_features_full.csv"  # tiny (717KB), used in full even for judge-mode
GENOMIC_PATH = "sample_data/sample_C1.1_100rows.csv"

pheno = di.load_phenotype(PHENO_PATH)
env = di.load_environmental(ENV_PATH)
genomic = di.load_genomic_population(GENOMIC_PATH)

print("=== Phenotype sample ===")
print(pheno.shape, "rows x cols")
print("YEAR range:", pheno["YEAR"].min(), "-", pheno["YEAR"].max())
print("LOCs:", pheno["LOC"].unique())
print("CLUSTER values:", pheno["CLUSTER"].unique())
print("Unique LINE_UNIQUE_ID populations:", pheno["LINE_UNIQUE_ID"].str.extract(r"^(C\d+\.\d+)\.")[0].unique())
print("Missing YLD_BE:", pheno["YLD_BE"].isna().sum(), "/", len(pheno))

print("\n=== Environment sample ===")
print(env.shape)
print(env[["YEAR", "LOC"]].drop_duplicates())

print("\n=== Phenotype <-> Environment merge ===")
pheno_env = di.merge_phenotype_environment(pheno, env)
matched = pheno_env["X04_PRCP"].notna().sum() if "X04_PRCP" in pheno_env else "col missing"
print(f"Rows with environmental match: {matched} / {len(pheno_env)}")

print("\n=== Genomic sample ===")
parents, progeny = di.split_genomic_parents_progeny(genomic)
print("Parents:", list(parents.index))
print("Progeny count:", len(progeny), "| markers:", genomic.shape[1])
print("Missing genotype calls (NA) total:", genomic.isna().sum().sum())

print("\n=== Phenotype <-> Genomic merge (population C1.1) ===")
pop1 = pheno[pheno["LINE_UNIQUE_ID"].str.match(r"^C1\.1\.")]
print("Phenotype rows for C1.1:", len(pop1))
merged, parents = di.merge_phenotype_genomic_for_population(pop1, genomic)
n_matched = merged["line_number_marker" if "line_number_marker" in merged else "line_number"].notna().sum()
marker_cols = [c for c in merged.columns if c.startswith("M0")]
n_with_markers = merged[marker_cols[0]].notna().sum() if marker_cols else 0
print(f"Phenotype rows successfully matched to a genomic marker row: {n_with_markers} / {len(merged)}")

if pop1["CROSS"].notna().any():
    cross_val = pop1["CROSS"].dropna().iloc[0]
    ok = di.verify_parents_match_cross(parents, cross_val)
    print(f"\nCROSS field '{cross_val}' matches genomic parent rows {list(parents.index)}: {ok}")
