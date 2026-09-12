"""
Baseline models required by the brief: something to prove the genomic model
is actually adding value over what a breeder could already do without it.
"""

import pandas as pd


def environmental_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 trait: str = "YLD_BE") -> pd.Series:
    """Predict every test observation with its environment's (YEAR, LOC)
    mean yield from the training data. No knowledge of the line at all --
    the floor a model has to beat."""
    env_means = train_df.groupby(["YEAR", "LOC"])[trait].mean()
    overall_mean = train_df[trait].mean()
    preds = test_df.set_index(["YEAR", "LOC"]).index.map(env_means)
    return pd.Series(preds, index=test_df.index).fillna(overall_mean)


def phenotypic_mean_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                              adj_col: str = "YLD_BE_ADJ") -> pd.Series:
    """Predict a line's environment-adjusted yield using its own mean
    adjusted performance from OTHER environments in the training data --
    the 'phenotypic BLUP' a breeder uses today without any genomics.
    Only works for lines with prior phenotype history; genomic prediction's
    whole value proposition is covering lines this baseline can't touch."""
    line_means = train_df.groupby("LINE_UNIQUE_ID")[adj_col].mean()
    overall_mean = train_df[adj_col].mean()
    preds = test_df["LINE_UNIQUE_ID"].map(line_means)
    return preds.fillna(overall_mean)
