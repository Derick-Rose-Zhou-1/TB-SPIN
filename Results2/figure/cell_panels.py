from __future__ import annotations
"""Draw the primary and external UMAP pairs and cell-type fraction and composition panels."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from recompute import BASE, OUT

PANELS = OUT / "cell_panels"
COLORS = {
    "LTBI": "#4E79A7",
    "ATB": "#D96A62",
    "neutral": "#B8B8B8",
    "gray": "#CECECE",
    "B cell": "#F28E2B",
    "Monocyte": "#E15759",
    "NK cell": "#4E79A7",
    "Neutrophil": "#59A14F",
    "T cell": "#76B7B2",
    "pDC": "#B07AA1",
}


def style(font_size: float = 9.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": font_size,
            "font.weight": "bold",
            "axes.titlesize": font_size + 1.0,
            "axes.titleweight": "bold",
            "axes.labelsize": font_size,
            "axes.labelweight": "bold",
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def bold_ticks(ax: plt.Axes) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: str) -> Path:
    pdf = PANELS / f"{stem}.pdf"
    png = PANELS / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return pdf


def add_umap_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    origin = (x0 + 0.035 * dx, y0 + 0.035 * dy)
    ax.annotate("", xy=(origin[0] + 0.16 * dx, origin[1]), xytext=origin,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"), clip_on=False)
    ax.annotate("", xy=(origin[0], origin[1] + 0.17 * dy), xytext=origin,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black"), clip_on=False)
    ax.text(origin[0] + 0.08 * dx, origin[1] - 0.04 * dy, xlabel,
            ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.text(origin[0] - 0.03 * dx, origin[1] + 0.085 * dy, ylabel,
            ha="right", va="center", rotation=90, fontsize=8.5, fontweight="bold")


def plot_umap_pair(data: pd.DataFrame, stem: str, external: bool) -> Path:
    style(9.5)
    fig, axes = plt.subplots(1, 2, figsize=(4.15, 2.2), constrained_layout=False)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.985, wspace=0.04)
    order = ["T cell", "NK cell", "B cell", "Monocyte", "Neutrophil", "pDC"]
    ax = axes[0]
    for label in order:
        part = data.loc[data["celltype_raw"].eq(label)]
        ax.scatter(part["UMAP_1"], part["UMAP_2"], s=0.11, alpha=0.74,
                   c=COLORS[label], linewidths=0, rasterized=True)
    fixed_internal = {
        "T cell": (-9.2, -2.2), "NK cell": (-4.0, 6.3), "B cell": (1.9, -13.3),
        "Monocyte": (10.1, 3.7), "Neutrophil": (3.2, -2.3), "pDC": (8.5, -3.0),
    }
    fixed_external = {
        "B cell": (-9.6, 5.9), "pDC": (-2.8, 8.0), "Monocyte": (4.7, 7.9),
        "T cell": (-2.7, -3.0), "NK cell": (5.4, -4.4),
    }
    positions = fixed_external if external else fixed_internal
    for label in order:
        part = data.loc[data["celltype_raw"].eq(label)]
        if part.empty:
            continue
        x, y = positions.get(label, tuple(part[["UMAP_1", "UMAP_2"]].median()))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.2,
                fontweight="bold", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_aspect("equal")
    ax.axis("off")
    add_umap_axes(ax, "UMAP 1", "UMAP 2")

    ax = axes[1]
    background = data.loc[data["risk_label"].eq("Other cells")]
    risk = data.loc[~data["risk_label"].eq("Other cells")]
    ax.scatter(background["UMAP_1"], background["UMAP_2"], s=0.09, alpha=0.53,
               c=COLORS["gray"], linewidths=0, rasterized=True)
    for risk_name, part in risk.groupby("risk_label"):
        is_infection = "infection" in str(risk_name).lower()
        ax.scatter(
            part["UMAP_1"], part["UMAP_2"],
            s=0.34 if is_infection else 0.18,
            alpha=0.98 if is_infection else 0.86,
            c="#2F6EA5" if is_infection else "#E15759",
            edgecolors="#1F4F78" if is_infection else "none",
            linewidths=0.10 if is_infection else 0,
            rasterized=True,
        )
    external_risk_positions = {
        "TB disease risk cells": (4.4, -2.25),
        "TB infection risk cells": (4.4, -4.75),
    }
    internal_risk_positions = {
        "TB infection risk cells": (-2.8, 6.1),
        "TB disease risk cells": (-8.2, 1.2),
    }
    for label, part in risk.groupby("risk_label"):
        if part.empty:
            continue
        positions = external_risk_positions if external else internal_risk_positions
        x, y = positions.get(str(label), tuple(part[["UMAP_1", "UMAP_2"]].median()))
        display = str(label).replace("TBINF", "TB infection").replace("ATB", "TB disease")
        ax.text(x, y, display, ha="center", va="center", fontsize=7.6,
                fontweight="bold", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_aspect("equal")
    ax.axis("off")
    add_umap_axes(ax, "UMAP 1", "UMAP 2")
    return save_figure(fig, stem)


def build_fraction_composition() -> Path:
    style(8.6)
    base = BASE / "p95"
    fraction = pd.read_csv(base / "FIG2_RISK_CELL_FRACTION_SOURCE.tsv", sep="\t")
    composition = pd.read_csv(base / "FIG2_RISK_CELL_COMPOSITION_SOURCE.tsv", sep="\t")
    stages = ["LTBI", "ATB"]
    display_stages = ["TB infection", "TB disease"]
    lineages = ["Monocyte", "B cell", "T cell", "NK cell", "Neutrophil"]
    fig = plt.figure(figsize=(4.15, 2.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.34, 0.80], wspace=0.54)
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(lineages))
    width = 0.30
    for offset, stage_name, display_stage in zip((-width / 2, width / 2), stages, display_stages):
        part = fraction.set_index(["display_stage", "celltype_raw"]).loc[stage_name]
        values = np.array([part.loc[cell, "risk_fraction_pct"] for cell in lineages])
        counts = np.array([part.loc[cell, "risk_n"] for cell in lineages])
        bars = ax.bar(x + offset, values, width=width, label=display_stage,
                      color=COLORS[stage_name], edgecolor="black", linewidth=0.55)
        for bar, value, count in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45,
                    f"{int(count)}\n{value:.1f}%", ha="center", va="bottom",
                    fontsize=5.7, fontweight="bold", linespacing=0.88)
    ax.set_title("Risk-cell fraction (%)", loc="left", fontsize=9.1, pad=15)
    ax.set_xticks(x, lineages, rotation=45, ha="right")
    ax.set_ylim(0, 31)
    ax.set_yticks([0, 10, 20, 30])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=2,
              handlelength=1.0, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    bold_ticks(ax)

    ax = fig.add_subplot(grid[0, 1])
    bottom = np.zeros(2)
    for lineage in ["NK cell", "T cell", "B cell", "Monocyte", "Neutrophil"]:
        values = []
        for stage_name in stages:
            row = composition.loc[(composition["display_stage"] == stage_name)
                                  & (composition["celltype_raw"] == lineage)].iloc[0]
            values.append(row["composition_pct"])
        bars = ax.bar(np.arange(2), values, bottom=bottom, width=0.52,
                      color=COLORS[lineage], edgecolor="black", linewidth=0.5, label=lineage)
        for idx, (bar, value) in enumerate(zip(bars, values)):
            if value >= 4.5:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[idx] + value / 2,
                        f"{value:.1f}%", ha="center", va="center", color="white",
                        fontsize=9.2, fontweight="bold")
        bottom += np.asarray(values)
    ax.set_title("Composition (%)", loc="left", fontsize=10.5, pad=17)
    ax.set_xticks([0, 1], display_stages, rotation=45, ha="right")
    ax.set_ylim(0, 100.5)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.50), handlelength=0.9, handletextpad=0.35)
    bold_ticks(ax)
    fig.subplots_adjust(left=0.09, right=0.84, bottom=0.24, top=0.76)
    return save_figure(fig, "CD_RISK_FRACTION_COMPOSITION_LARGE_FONT")

if __name__ == "__main__":
    PANELS.mkdir(parents=True, exist_ok=True)
    for site in ("INTERNAL", "EXTERNAL"):
        frame = pd.read_csv(BASE / "p95" / f"FIG2_{site}_FINAL_P95_UMAP_SOURCE.tsv.gz", sep="\t")
        plot_umap_pair(frame, site, external=(site == "EXTERNAL"))
    build_fraction_composition()
