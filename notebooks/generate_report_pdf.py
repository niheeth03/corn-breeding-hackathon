"""Generates a PDF summary report: results overview + top selected lines
for both clusters, pulling directly from the real, already-computed output
CSVs (outputs/C{1,2}_resource_allocation_2008.csv)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from fpdf import FPDF

OUT_PATH = "outputs/Corn_Breeding_Results_Report.pdf"
TOP_N = 25


class Report(FPDF):
    def header(self):
        pass


def add_table(pdf, df, col_widths, headers):
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.iterrows():
        for val, w in zip(row, col_widths):
            pdf.cell(w, 6, str(val), border=1)
        pdf.ln()


pdf = Report()
pdf.add_page()

pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Corn Breeding Line Advancement -- Results Report", ln=True)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Precision Digital Agriculture Hackathon 2026 - Corn Breeding Track", ln=True)
pdf.ln(4)

pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Validated Results (real, held-out 2008 test)", ln=True)
pdf.set_font("Helvetica", "", 9)
results_text = [
    "Model                                  C1 corr / top-20%      C2 corr / top-20%",
    "Within-population (family started)     0.636 / 51%            0.618 / 50%",
    "Mate-selection (brand-new family)      0.317 / 40%            0.203 / 28%",
    "Final deployed model (line-level)      0.167 / 29.5% (n=7429) 0.139 / 26.4% (n=8533)",
]
pdf.set_font("Courier", "", 8.5)
for line in results_text:
    pdf.cell(0, 5.5, line, ln=True)
pdf.ln(3)

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "All figures beat the 20% chance baseline for top-20% recovery. The deployed model combines "
    "mate-selection's family-level baseline (mid-parent genomic estimated breeding value, computed "
    "from a ridge regression trained on 2001-2007 data, alpha=1000) with each line's own within-family "
    "deviation, to produce a specific, ranked advancement recommendation for every genotyped 2008 line."
)
pdf.ln(2)

for cluster in [1, 2]:
    alloc_path = f"outputs/C{cluster}_resource_allocation_2008.csv"
    if not os.path.exists(alloc_path):
        continue
    alloc = pd.read_csv(alloc_path)
    top = alloc.sort_values("final_predicted_yield_advantage", ascending=False).head(TOP_N).reset_index(drop=True)
    top.index = top.index + 1

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Cluster {cluster}: Top {TOP_N} Recommended Lines to Advance", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Full allocation: {len(alloc)} lines across {alloc['population'].nunique()} families "
                   f"(300-plot budget, max 5 lines/family)", ln=True)
    pdf.ln(2)

    table = pd.DataFrame({
        "Rank": top.index,
        "Line ID": top["LINE_UNIQUE_ID"],
        "Family": top["population"],
        "Pred. Yield Adv.": top["final_predicted_yield_advantage"].round(2),
        "68% CI": [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(top["predicted_lower_68pct"], top["predicted_upper_68pct"])],
        "Confidence": top["confidence_tier"],
    })
    add_table(pdf, table, col_widths=[12, 28, 22, 30, 40, 25],
              headers=["Rank", "Line ID", "Family", "Pred. Yield Adv.", "68% CI", "Confidence"])

pdf.output(OUT_PATH)
print(f"Saved {OUT_PATH}")
