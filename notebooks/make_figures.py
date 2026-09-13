"""Generates presentation-ready result figures for the hackathon demo,
using the validated dataviz-skill palette. Run from repo root."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- validated palette (light mode) ----
BLUE = "#2a78d6"      # slot 1 -- Cluster 1
ORANGE = "#eb6834"    # slot 2 -- Cluster 2
AQUA = "#1baf7a"       # slot 3 -- High confidence / good
YELLOW = "#eda100"     # Medium confidence
RED = "#e34948"        # Low confidence / critical contrast
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY, "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.grid": True,
    "grid.color": GRIDLINE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11,
})

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)


def style_ax(ax, hide_x_spine=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if hide_x_spine:
        ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


# ============================================================
# FIGURE 1: The rigor story -- naive CV vs honest CV
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
schemes = ["Naive random CV\n(leaky)", "Leave-population-out\n(brand-new family)", "Within-population\n(family already tested)"]
c1_vals = [0.565, -0.117, 0.636]
c2_vals = [0.543, 0.176, 0.618]
x = np.arange(len(schemes))
w = 0.32
b1 = ax.bar(x - w/2, c1_vals, width=w, color=BLUE, label="Cluster 1", zorder=3)
b2 = ax.bar(x + w/2, c2_vals, width=w, color=ORANGE, label="Cluster 2", zorder=3)
ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4 if h >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK_PRIMARY)
ax.set_xticks(x); ax.set_xticklabels(schemes, fontsize=10)
ax.set_ylabel("Correlation (predicted vs. actual)")
ax.set_title("The naive validation number is fake -- honest testing changes the story", fontsize=13, fontweight="bold", pad=42, loc="left")
ax.text(0, -0.28, "Naive CV looks strong only because it leaks sibling information across the train/test split.", fontsize=9.5, color=INK_SECONDARY, transform=ax.transAxes)
ax.set_ylim(-0.18, 0.72)
style_ax(ax)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/1_naive_vs_honest_cv.png", facecolor=SURFACE)
plt.close(fig)


# ============================================================
# FIGURE 2: Our one model, two views -- family-level and line-level
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
models = ["Family-level baseline\n(mid-parent GEBV)", "Final output\n(individual line ranking)"]
c1_rec = [40, 29.5]
c2_rec = [28, 26.4]
x = np.arange(len(models))
w = 0.32
b1 = ax.bar(x - w/2, c1_rec, width=w, color=BLUE, label="Cluster 1", zorder=3)
b2 = ax.bar(x + w/2, c2_rec, width=w, color=ORANGE, label="Cluster 2", zorder=3)
ax.axhline(20, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
ax.text(1.48, 20.6, "chance = 20%", fontsize=9, color=INK_MUTED, ha="right")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0f}%", (bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
ax.set_xlim(-0.55, 1.55)
ax.set_ylabel("Top-20% recovery (real 2008 holdout)")
ax.set_ylim(0, 46)
ax.set_title("Our model beats random selection at both stages, on real 2008 data", fontsize=13, fontweight="bold", pad=14, loc="left")
style_ax(ax)
ax.legend(frameon=False, loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/2_model_comparison.png", facecolor=SURFACE)
plt.close(fig)


# ============================================================
# FIGURE 3: Lift / enrichment curve at different selection intensities
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
cutoffs = [5, 10, 20]
c1_curve = [12.9, 20.1, 29.5]
c2_curve = [8.0, 14.9, 26.4]
chance_line = [0, 100]
ax.plot(chance_line, chance_line, color=INK_MUTED, linewidth=1.6, linestyle=(0, (4, 3)), label="Chance (random selection)", zorder=2)
ax.plot(cutoffs, c1_curve, color=BLUE, linewidth=2.5, marker="o", markersize=7, label="Cluster 1", zorder=3)
ax.plot(cutoffs, c2_curve, color=ORANGE, linewidth=2.5, marker="o", markersize=7, label="Cluster 2", zorder=3)
for cx, cy in zip(cutoffs, c1_curve):
    ax.annotate(f"{cy:.0f}%", (cx, cy), xytext=(6, 6), textcoords="offset points", fontsize=9.5, color=BLUE)
for cx, cy in zip(cutoffs, c2_curve):
    ax.annotate(f"{cy:.0f}%", (cx, cy), xytext=(6, -14), textcoords="offset points", fontsize=9.5, color=ORANGE)
ax.set_xlim(0, 23); ax.set_ylim(0, 35)
ax.set_xticks(cutoffs); ax.set_xticklabels([f"Top {c}%" for c in cutoffs])
ax.set_xlabel("Selection intensity (how large a slice you can afford to fund)")
ax.set_ylabel("% of true best lines actually recovered")
ax.set_title("The tighter the budget, the bigger our real advantage over guessing", fontsize=13, fontweight="bold", pad=14, loc="left")
style_ax(ax)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/3_enrichment_curve.png", facecolor=SURFACE)
plt.close(fig)

# ============================================================
# FIGURE 4: Predicted vs actual, real 2008 holdout (both clusters)
# ============================================================
from src import genomic_prediction as gp
from src import final_recommendation as fr

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), dpi=200)
cluster_info = [
    (1, "outputs/merged_master_C1.parquet", "Simplified Hackathon Dataset V3/ImputedPopulationsC1", BLUE),
    (2, "outputs/merged_master_C2.parquet", "Simplified Hackathon Dataset V3/ImputedPopulationsC2", ORANGE),
]
rng = np.random.default_rng(0)

for ax, (cluster, parquet, gdir, color) in zip(axes, cluster_info):
    df = pd.read_parquet(parquet)
    marker_cols = gp.marker_columns(df)
    df = gp.add_environment_adjusted_yield(df, trait="YLD_BE")
    result = fr.build_ranked_recommendations(df, marker_cols, gdir, cluster=cluster, test_year=2008)
    result["base_id"] = result["LINE_UNIQUE_ID"].apply(lambda s: ".".join(s.split(".")[:3]))
    d2008 = df[df["YEAR"] == 2008].copy()
    d2008["base_id"] = d2008["LINE_UNIQUE_ID"].apply(lambda s: ".".join(s.split(".")[:3]))
    actual = d2008.groupby("base_id")["YLD_BE_ADJ"].mean()
    merged = result.set_index("base_id")[["final_predicted_yield_advantage"]].join(actual, how="inner").dropna()

    corr = np.corrcoef(merged["YLD_BE_ADJ"], merged["final_predicted_yield_advantage"])[0, 1]
    sample = merged.sample(n=min(1500, len(merged)), random_state=0)
    ax.scatter(sample["final_predicted_yield_advantage"], sample["YLD_BE_ADJ"], s=14, alpha=0.25, color=color, linewidths=0, zorder=3)
    m, b = np.polyfit(merged["final_predicted_yield_advantage"], merged["YLD_BE_ADJ"], 1)
    xs = np.linspace(merged["final_predicted_yield_advantage"].min(), merged["final_predicted_yield_advantage"].max(), 50)
    ax.plot(xs, m * xs + b, color=INK_PRIMARY, linewidth=2, zorder=4)
    ax.text(0.04, 0.94, f"r = {corr:.2f}\nn = {len(merged):,}", transform=ax.transAxes, fontsize=11,
            va="top", color=INK_PRIMARY, fontweight="bold")
    ax.set_title(f"Cluster {cluster}", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Predicted yield advantage")
    ax.set_ylabel("Actual yield advantage (2008, real)")
    style_ax(ax)

fig.suptitle("Model predictions track real 2008 outcomes", fontsize=14, fontweight="bold", x=0.02, ha="left", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/4_predicted_vs_actual.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("Figure 4 done.")

# ============================================================
# FIGURE 5: Top recommended lines with uncertainty (both clusters)
# ============================================================
from matplotlib.lines import Line2D

def make_top_picks_figure(cluster, fig_num):
    alloc = pd.read_csv(f"outputs/C{cluster}_resource_allocation_2008.csv")
    top = alloc.sort_values("final_predicted_yield_advantage", ascending=False).head(15).iloc[::-1].reset_index(drop=True)

    tier_color = {"High": AQUA, "Medium": YELLOW, "Low": RED}
    colors = top["confidence_tier"].map(tier_color)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    y = np.arange(len(top))
    xerr = np.vstack([
        top["final_predicted_yield_advantage"] - top["predicted_lower_68pct"],
        top["predicted_upper_68pct"] - top["final_predicted_yield_advantage"],
    ])
    ax.errorbar(top["final_predicted_yield_advantage"], y, xerr=xerr, fmt="none", ecolor=INK_MUTED, elinewidth=1.6, capsize=3, zorder=2)
    ax.scatter(top["final_predicted_yield_advantage"], y, s=90, color=colors, zorder=3, edgecolors=SURFACE, linewidths=1)
    ax.set_yticks(y)
    ax.set_yticklabels(top["LINE_UNIQUE_ID"], fontsize=9.5)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("Predicted yield advantage (with 68% confidence interval)")
    ax.set_title(f"Top 15 recommended lines to advance -- Cluster {cluster}", fontsize=13, fontweight="bold", loc="left", pad=14)
    style_ax(ax)
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=9, label=lbl) for lbl, c in tier_color.items()]
    ax.legend(handles=handles, title="Confidence", frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fig_num}_top_picks_uncertainty_c{cluster}.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"Figure {fig_num} (Cluster {cluster}) done.")


make_top_picks_figure(1, 5)
make_top_picks_figure(2, 6)
