from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import os
ROOT = Path(os.environ.get("RESULT6_INPUT", "inputs"))
SOURCE = ROOT / "COHORT_SEPARATED_METRICS.tsv"
OUT = ROOT.parent / "figures"
PDF = OUT / "FIG6_TB_SPIN_AUC_95CI.pdf"
PNG = OUT / "FIG6_TB_SPIN_AUC_95CI_600dpi.png"
SOURCE_COPY = OUT / "FIG6_SOURCE_DATA.tsv"

COHORTS = ["GSE79362", "GSE94438"]
ENDPOINTS = ["H1", "H2", "H3"]
DISPLAY = {
    "TBSTATE": "TB-SPIN",
    "SIDISH": "SIDISH",
    "SCAB": "scAB",
    "SCPAS": "scPAS",
    "DEGAS": "DEGAS",
}
COLORS = {
    "TBSTATE": "#D77A8A",
    "SIDISH": "#E5B45F",
    "SCAB": "#83B8A7",
    "SCPAS": "#A9D2EA",
    "DEGAS": "#D5A2BD",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, sep="\t")
    data["method"] = data["method"].astype(str).str.upper()
    # The requested manuscript comparison excludes Scissor and the combined
    # GSE107993/GSE107994 cohort. H2/H3 SIDISH estimates are not available.
    data = data[
        data["cohort"].isin(COHORTS)
        & data["method"].isin(DISPLAY)
        & ~((data["method"] == "SIDISH") & (data["endpoint"].isin(["H2", "H3"])))
    ].copy()
    data.to_csv(SOURCE_COPY, sep="\t", index=False)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.7), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.16, wspace=0.17, hspace=0.38)

    for row, cohort in enumerate(COHORTS):
        row_data = data[data["cohort"] == cohort]
        fig.text(
            0.014,
            0.952 if row == 0 else 0.485,
            "a" if row == 0 else "b",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.052,
            0.952 if row == 0 else 0.485,
            cohort,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        for col, endpoint in enumerate(ENDPOINTS):
            ax = axes[row, col]
            part = row_data[row_data["endpoint"] == endpoint].copy()
            methods = [m for m in ["TBSTATE", "SIDISH", "SCAB", "SCPAS", "DEGAS"] if m in set(part["method"])]
            # Stable x order is consistent across cohorts; missing estimates are omitted.
            part = part.set_index("method").reindex(methods).dropna(subset=["estimate"])
            x = np.arange(len(part))
            est = part["estimate"].to_numpy(dtype=float)
            low = part["ci_low"].to_numpy(dtype=float)
            high = part["ci_high"].to_numpy(dtype=float)
            colors = [COLORS[m] for m in part.index]

            ax.bar(x, est, color=colors, edgecolor="#4A4A4A", linewidth=0.45, width=0.68, zorder=2)
            ax.errorbar(
                x,
                est,
                yerr=np.vstack([est - low, high - est]),
                fmt="none",
                ecolor="#373737",
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=3,
            )
            for xi, value in zip(x, est):
                ax.text(xi, min(high[xi] + 0.035, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)

            ax.axhline(0.5, color="#9A9A9A", lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, 1.09)
            ax.set_xlim(-0.6, max(0.6, len(x) - 0.4))
            ax.set_xticks(x, [DISPLAY[m] for m in part.index], rotation=42, ha="right", rotation_mode="anchor")
            ax.set_title(endpoint, fontweight="bold", pad=7)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.45, zorder=0)
            ax.tick_params(axis="both", length=2.5, width=0.65, pad=2)
            if col == 0:
                ax.set_ylabel("AUC")
            else:
                ax.tick_params(labelleft=False)

    fig.savefig(PDF, bbox_inches="tight", facecolor="white", metadata={"Title": "TB-SPIN AUC by cohort and endpoint"})
    fig.savefig(PNG, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Compact provenance/coverage report: no metrics are recalculated here.
    expected = {(c, e) for c in COHORTS for e in ENDPOINTS}
    actual = set(data[["cohort", "endpoint"]].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError(f"Unexpected cohort-endpoint coverage: missing={sorted(expected-actual)}")
    print(f"PDF={PDF}")
    print(f"PNG={PNG}")
    print(f"SOURCE_DATA={SOURCE_COPY}")
    print(f"N_ROWS={len(data)}")
    print("SCISSOR_PLOTTED=NO")
    print("SIDISH_H2_H3_PLOTTED=NO")
    print("COMBINED_COHORT_PLOTTED=NO")
    print("METRICS_RECOMPUTED=NO")


if __name__ == "__main__":
    main()
