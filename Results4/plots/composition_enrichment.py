from pathlib import Path
import math
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import fisher_exact


BASE = Path(__file__).resolve().parents[1] / "inputs/FIGURE4_C05_COMPLETE_V15/source_tables"
OUT = Path(__file__).resolve().parents[1] / "panel_outputs"
FIG = OUT / "figures"
SRC = OUT / "source_tables"
AUD = OUT / "audits"
SCR = OUT / "scripts"
DOC = OUT / "docs"

for d in (FIG, SRC, AUD, SCR, DOC):
    d.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

LINEAGE_ORDER = ["B cell", "T cell", "NK cell", "Monocyte", "Neutrophil"]
LINEAGE_COLORS = {
    "B cell": "#A8842C",
    "T cell": "#2F6C74",
    "NK cell": "#6C8199",
    "Monocyte": "#A45A4A",
    "Neutrophil": "#8B3E36",
}

ROW_LABELS = {
    "C01": "M. tuberculosis\ninfection",
    "C02": "Incipient TB",
    "C03": "Asymptomatic TB\nvs infection",
    "C04": "Symptomatic TB\nvs infection",
    "C05_ASYM": "C05 asymptomatic-\nassociated",
    "C05_SYM": "C05 symptomatic-\nassociated",
}

BAR_LABELS = [
    ("C01", "M. tuberculosis\ninfection"),
    ("C02", "Incipient\nTB"),
    ("C03", "Asymptomatic\nTB"),
    ("C04", "Symptomatic\nTB"),
    ("C05_ASYM", "C05 asymptomatic-\nassociated"),
    ("C05_SYM", "C05 symptomatic-\nassociated"),
]


def clean_term(term):
    term = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term))
    term = term.replace("MHC Class Ii", "MHC class II")
    term = term.replace("B Cell", "B cell")
    term = term.replace("T Cell", "T cell")
    return term


def wrap_label(text, width=30):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def save_fig(fig, stem, png=True):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    if png:
        fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out


def make_composition():
    comp = pd.read_csv(BASE / "R4_SCPAS_COMPOSITION.tsv", sep="\t")
    c05 = pd.read_csv(BASE / "02_C05_LINEAGE_COMPOSITION.tsv", sep="\t")

    rows = []
    for code in ["C01", "C02", "C03", "C04"]:
        sub = comp[(comp["contrast"] == code) & (comp["association_label"] == "scPAS+")]
        total = int(sub["label_total_n"].iloc[0])
        for lineage in LINEAGE_ORDER:
            hit = sub[sub["celltype"] == lineage]
            n = int(hit["cell_n"].iloc[0]) if not hit.empty else 0
            rows.append({"group": code, "lineage": lineage, "n": n, "total": total})

    map_assoc = {
        "asymptomatic_associated_reference": "C05_ASYM",
        "symptomatic_associated_target": "C05_SYM",
    }
    for assoc, group in map_assoc.items():
        sub = c05[c05["association"] == assoc]
        total = int(round(sub["associated_cells"].sum()))
        # The formal C05 table contains minor populations too; denominator is all associated cells.
        for lineage in LINEAGE_ORDER:
            hit = sub[sub["major_population"] == lineage]
            n = int(hit["associated_cells"].iloc[0]) if not hit.empty else 0
            rows.append({"group": group, "lineage": lineage, "n": n, "total": total})

    out = pd.DataFrame(rows)
    out["fraction"] = out["n"] / out["total"]
    out.to_csv(AUD / "C05_CELL_COMPOSITION_AUDIT.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    x = np.arange(len(BAR_LABELS))
    bottom = np.zeros(len(BAR_LABELS))
    group_to_idx = {g: i for i, (g, _) in enumerate(BAR_LABELS)}

    for lineage in LINEAGE_ORDER:
        vals = []
        for group, _ in BAR_LABELS:
            sub = out[(out["group"] == group) & (out["lineage"] == lineage)]
            vals.append(float(sub["fraction"].iloc[0]) if not sub.empty else 0.0)
        ax.bar(x, np.array(vals) * 100, bottom=bottom * 100, color=LINEAGE_COLORS[lineage],
               width=0.72, label=lineage, edgecolor="white", linewidth=0.4)
        for i, v in enumerate(vals):
            if v >= 0.10:
                ax.text(i, (bottom[i] + v / 2) * 100, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=6.5, color="white")
        bottom += np.array(vals)

    for group, label in BAR_LABELS:
        i = group_to_idx[group]
        total = int(out[out["group"] == group]["total"].iloc[0])
        ax.text(i, 103, f"n={total:,}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 112)
    ax.set_ylabel("Associated cells (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in BAR_LABELS], rotation=35, ha="right")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=6.5,
              handlelength=1.0, columnspacing=0.9)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.36)
    save_fig(fig, "Fig4c_six_group_composition")
    return out


def make_enrichment():
    old = pd.read_csv(BASE / "R4_CELLTYPE_ENRICHMENT.tsv", sep="\t")
    old = old[(old["contrast"].isin(["C01", "C02", "C03", "C04"])) & (old["celltype"].isin(LINEAGE_ORDER))].copy()
    old["group"] = old["contrast"]
    old = old.rename(columns={"celltype": "lineage", "OR": "odds_ratio", "FDR": "BH_FDR", "PASplus_n": "associated_n"})
    old["log2_OR"] = np.log2(old["odds_ratio"].replace(0, np.nan))
    old["source"] = "R4_CELLTYPE_ENRICHMENT.tsv"
    old_keep = old[["group", "lineage", "associated_n", "odds_ratio", "log2_OR", "P", "BH_FDR", "source"]]

    c05_comp = pd.read_csv(BASE / "02_C05_LINEAGE_COMPOSITION.tsv", sep="\t")
    c05_sum = pd.read_csv(BASE / "02_C05_DIRECT_SYMPTOM_AXIS.tsv", sep="\t")
    total_cells = 356949
    asym_total = int(c05_sum.loc[c05_sum["orientation"] == "asymptomatic_associated_reference", "cells"].iloc[0])
    sym_total = int(c05_sum.loc[c05_sum["orientation"] == "symptomatic_associated_target", "cells"].iloc[0])
    bg_total = int(c05_sum.loc[c05_sum["orientation"] == "background", "cells"].iloc[0])

    assoc_map = {
        "C05_ASYM": "asymptomatic_associated_reference",
        "C05_SYM": "symptomatic_associated_target",
    }
    records = []
    for group, assoc in assoc_map.items():
        sub = c05_comp[c05_comp["association"] == assoc]
        other_assoc = "symptomatic_associated_target" if assoc == "asymptomatic_associated_reference" else "asymptomatic_associated_reference"
        other = c05_comp[c05_comp["association"] == other_assoc]
        assoc_total = asym_total if group == "C05_ASYM" else sym_total
        pvals = []
        tmp = []
        for lineage in LINEAGE_ORDER:
            a = int(sub.loc[sub["major_population"] == lineage, "associated_cells"].sum())
            total_lineage = int(sub.loc[sub["major_population"] == lineage, "total_cells_in_population"].max())
            if total_lineage == 0:
                total_lineage = int(other.loc[other["major_population"] == lineage, "total_cells_in_population"].max())
            other_a = int(other.loc[other["major_population"] == lineage, "associated_cells"].sum())
            c = int(total_lineage - a - other_a)
            b = assoc_total - a
            d = bg_total - c
            table = np.array([[a, b], [c, d]])
            odds, p = fisher_exact(table, alternative="two-sided")
            if odds == 0:
                log2_or = -5.0
            elif math.isinf(odds):
                log2_or = 5.0
            else:
                log2_or = math.log2(odds)
            pvals.append(p)
            tmp.append({
                "group": group, "lineage": lineage, "associated_n": assoc_total,
                "associated_lineage_n": a, "background_n": bg_total,
                "background_lineage_n": c, "odds_ratio": odds, "log2_OR": log2_or,
                "P": p, "source": "02_C05_LINEAGE_COMPOSITION.tsv"
            })
        fdr = bh_fdr(pvals)
        for rec, q in zip(tmp, fdr):
            rec["BH_FDR"] = q
            records.append(rec)

    c05_enr = pd.DataFrame(records)
    c05_enr.to_csv(AUD / "FIG4D_C05_ENRICHMENT_AUDIT.tsv", sep="\t", index=False)
    all_enr = pd.concat([old_keep, c05_enr[old_keep.columns]], ignore_index=True)
    all_enr.to_csv(SRC / "Fig4d_lineage_enrichment_source_used.tsv", sep="\t", index=False)

    mat = all_enr.pivot(index="group", columns="lineage", values="log2_OR").reindex([g for g, _ in BAR_LABELS])[LINEAGE_ORDER]
    qmat = all_enr.pivot(index="group", columns="lineage", values="BH_FDR").reindex([g for g, _ in BAR_LABELS])[LINEAGE_ORDER]
    nmap = all_enr.groupby("group")["associated_n"].first().to_dict()

    fig, ax = plt.subplots(figsize=(5.1, 3.35))
    cmap = LinearSegmentedColormap.from_list("r4", ["#6C8199", "#F4EFE5", "#A45A4A"])
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(LINEAGE_ORDER)))
    ax.set_xticklabels(LINEAGE_ORDER, rotation=45, ha="right", rotation_mode="anchor")
    labels = [f"{ROW_LABELS[g]}\n(n={int(nmap.get(g, 0)):,})" for g, _ in BAR_LABELS]
    ax.set_yticks(np.arange(len(BAR_LABELS)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            q = qmat.iloc[i, j]
            stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
            if stars:
                ax.text(j, i, stars, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#202020")
    for x in np.arange(-.5, len(LINEAGE_ORDER), 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-.5, len(BAR_LABELS), 1):
        ax.axhline(y, color="white", lw=0.8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log2 odds ratio", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.34, right=0.90, bottom=0.23, top=0.98)
    save_fig(fig, "Fig4d_six_group_lineage_enrichment")
    return all_enr


if __name__ == "__main__":
    make_composition()
    make_enrichment()
