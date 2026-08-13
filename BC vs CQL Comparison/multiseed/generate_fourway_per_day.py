"""Reconstruct the report's Figure 2 (fourway_multiseed_per_day.png).

The original chart had no generating script in the repo -- it was assembled
by hand outside version control. This script rebuilds it from the exact
per-day, 30-seed mean +/- std values already verified and reported in the
paper's own Appendix A.3 tables (sourcecode/09_supplementary_information.tex:
tab:multiseed-bcreg-per-seed, tab:multiseed-bccirl-per-seed,
tab:multiseed-cqlreg-per-seed, tab:multiseed-per-seed), so no rollout re-run
is needed -- only re-plotting with larger label/legend text.

Run from this folder: python generate_fourway_per_day.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

DAYS = ["16 June", "17 June", "18 June", "19 June"]

# mean +/- std over 30 seeds, per day, transcribed from the report's own
# Appendix A.3 tables (values already verified in-session against the
# underlying rollout data).
MEAN = {
    "BC Regular":  [1.510, 1.953, 2.063, 0.817],
    "BC CIRL":     [1.286, 1.482, 1.796, 0.331],
    "CQL Regular": [1.354, 1.553, 1.760, 0.807],
    "CQL CIRL":    [0.320, 0.334, 0.457, 0.058],
}
STD = {
    "BC Regular":  [0.358, 0.482, 0.454, 0.430],
    "BC CIRL":     [0.000, 0.000, 0.000, 0.000],
    "CQL Regular": [0.537, 0.797, 0.855, 0.386],
    "CQL CIRL":    [0.008, 0.009, 0.012, 0.002],
}
EXPERT = [1.286, 1.482, 1.796, 0.331]

ORDER = ["BC Regular", "BC CIRL", "CQL Regular", "CQL CIRL"]
COLORS = {
    "BC Regular":  "#8fd18f",
    "BC CIRL":     "#2ca02c",
    "CQL Regular": "#f4a582",
    "CQL CIRL":    "#d62728",
}

x = np.arange(len(DAYS))
n_bars = len(ORDER) + 1  # + Expert PI
width = 0.8 / n_bars

fig, ax = plt.subplots(figsize=(11, 6.5))

# Expert PI first (leftmost of each group)
ax.bar(x - 0.4 + width * 0.5, EXPERT, width, label="Expert PI",
       color="#888888", alpha=0.85)

for i, name in enumerate(ORDER, start=1):
    ax.bar(x - 0.4 + width * (i + 0.5), MEAN[name], width,
           yerr=STD[name], capsize=4, label=name,
           color=COLORS[name], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(DAYS, fontsize=14)
ax.set_ylabel("Zero-shot MAE (C)", fontsize=16)
ax.set_title("Zero-shot MAE per unseen day, 30 seeds (mean +/- std),\n"
             "all four offline controllers vs. Expert PI", fontsize=17, fontweight="bold")
ax.tick_params(axis="both", labelsize=13)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=13, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fourway_multiseed_per_day.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved -> {out_path}")
plt.close(fig)
