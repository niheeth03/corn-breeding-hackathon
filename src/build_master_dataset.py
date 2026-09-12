"""
Build the master (phenotype + environment + genomic) feature table for a
cluster, streaming one population at a time so memory stays bounded even
though the full C1/C2 datasets are 500k+ rows x ~3000 columns if flattened.

Usage:
  python src/build_master_dataset.py --mode sample
  python src/build_master_dataset.py --mode full --cluster 1 --limit 10
  python src/build_master_dataset.py --mode full --cluster 1
"""

from __future__ import annotations

import argparse
import time
import sys
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data_integration as di

SAMPLE_PHENO = "sample_data/sample_C1_phenotype_100rows.csv"
SAMPLE_ENV = "sample_data/environmental_features_full.csv"
SAMPLE_GENOMIC_DIR = "sample_data"  # sample_C1.1_100rows.csv lives here directly

FULL_DATA_DIR = "Simplified Hackathon Dataset V3"
FULL_ENV = f"{FULL_DATA_DIR}/environmental_features.csv"


def _genomic_path_for_sample(cluster, pop):
    # the single provided sample genomic file is named sample_C1.1_100rows.csv
    return f"{SAMPLE_GENOMIC_DIR}/sample_C{cluster}.{pop}_100rows.csv"


def run(mode: str, cluster: int, limit: int | None, out_path: str):
    t0 = time.time()

    if mode == "sample":
        pheno = di.load_phenotype(SAMPLE_PHENO)
        env = di.load_environmental(SAMPLE_ENV)
        genomic_dir = SAMPLE_GENOMIC_DIR
        # the sample genomic file doesn't follow the standard naming pattern,
        # so patch iter_merged_populations' expected path just for this one file
        import types
        pheno_env = di.merge_phenotype_environment(pheno, env)

        def _iter():
            genomic = di.load_genomic_population("sample_data/sample_C1.1_100rows.csv")
            pe = di.prepare_full_feature_table(pheno_env)
            parsed = pe["LINE_UNIQUE_ID"].apply(di.parse_line_unique_id)
            pe = pe.assign(line_number=parsed.apply(lambda t: t[2] if t else None))
            pe = pe[pe["LINE_UNIQUE_ID"].str.match(r"^C1\.1\.")]
            _, progeny = di.split_genomic_parents_progeny(genomic)
            marker_cols = list(genomic.columns)
            progeny[marker_cols] = progeny[marker_cols].astype("float32")
            merged = pe.merge(progeny, on="line_number", how="left", suffixes=("", "_marker"))
            yield 1, merged, len(marker_cols)

        pop_iter = _iter()
    else:
        pheno_path = f"{FULL_DATA_DIR}/C{cluster}_Phenotype_Data_V2.csv"
        genomic_dir = f"{FULL_DATA_DIR}/ImputedPopulationsC{cluster}"
        print(f"Loading full phenotype file: {pheno_path} ...")
        pheno = di.load_phenotype(pheno_path)
        env = di.load_environmental(FULL_ENV)
        pheno_env = di.merge_phenotype_environment(pheno, env)
        print(f"  -> {len(pheno_env):,} phenotype rows loaded and joined to environment "
              f"in {time.time() - t0:.1f}s")

        populations = None
        if limit:
            all_pops = sorted(pheno_env["LINE_UNIQUE_ID"]
                               .str.extract(r"^C\d+\.(\d+)\.")[0].dropna().astype(int).unique())
            populations = all_pops[:limit]

        pop_iter = di.iter_merged_populations(pheno_env, genomic_dir, cluster, populations)

    writer = None
    total_rows = 0
    total_pops = 0
    n_markers = None

    for pop_id, merged, n_markers in pop_iter:
        table = pa.Table.from_pandas(merged, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total_rows += len(merged)
        total_pops += 1
        if total_pops % 50 == 0:
            print(f"  ... {total_pops} populations processed, {total_rows:,} rows so far "
                  f"({time.time() - t0:.1f}s elapsed)")

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    print("\n=== Build complete ===")
    print(f"Populations processed : {total_pops}")
    print(f"Total merged rows     : {total_rows:,}")
    print(f"Markers per row       : {n_markers}")
    print(f"Output file           : {out_path}")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Output file size      : {size_mb:.1f} MB")
    print(f"Elapsed time          : {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--cluster", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N populations (full mode)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    default_out = (
        f"outputs/merged_master_sample.parquet"
        if args.mode == "sample"
        else f"outputs/merged_master_C{args.cluster}.parquet"
    )
    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    run(args.mode, args.cluster, args.limit, out_path)
