"""Source-locked Figure 4 assembly matching the supplied six-panel layout.

Only plot rendering and page geometry change. UMAP coordinates, state labels,
cell counts, composition fractions, enrichment estimates, and pathway charts
are read from frozen local sources. Dense UMAP point layers are rasterized at
450 dpi; all labels, c/d marks, the clinical schematic, and pathway panels
remain PDF vectors.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1] / "inputs"
OLD = ROOT / "FIGURE4_C05_COMPLETE_V15"
OUT = ROOT / "FIG4_RESULT4_FINAL_AUDITED_20260925"
SRC = OLD / "source_tables"
FROZEN_BOTTOM = ROOT / "FIG4_C05_SOURCE_VERIFIED_V13_20260912" / "01_source_locked"
BOTTOM = OUT / "bottom_redraw" / "03_figures" / "Fig4_bottom_new_editable.pdf"
CLINICAL_PPT = ROOT / "FIG4A_CLINICAL_CONTRASTS_PPT" / "FIG4A_CLINICAL_CONTRASTS_EDITABLE.pptx"
MANUSCRIPT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

LINEAGES = ["B cell", "T cell", "NK cell", "Monocyte", "Neutrophil"]
COLORS = {"B cell": "#A8842C", "T cell": "#2F6C74", "NK cell": "#6C8199", "Monocyte": "#A45A4A", "Neutrophil": "#8B3E36"}
GROUPS = ["C01", "C02", "C03", "C04", "C05_ASYM", "C05_SYM"]
GROUP_DISPLAY = [
    "C01 TB infection",
    "C02 Incipient TB",
    "C03 Asymptomatic TB",
    "C04 Symptomatic TB",
    "C05− asymptomatic-\nassociated",
    "C05+ symptomatic-\nassociated",
]
UMAP_SPECS = [
    ("TB infection", "C01_ALIGNED_LABELS.tsv.gz", "positive", "#1F6F8B", 1581),
    ("Incipient TB", "C02_ALIGNED_LABELS.tsv.gz", "positive", "#8C6BB1", 2034),
    ("Asymptomatic TB", "C03_ALIGNED_LABELS.tsv.gz", "positive", "#C49A2C", 5010),
    ("Symptomatic TB", "C04_ALIGNED_LABELS.tsv.gz", "positive", "#C65A46", 3470),
    ("asymptomatic-\nassociated", "C05_ALIGNED_LABELS.tsv.gz", "negative", "#A8842C", 2250),
    ("symptomatic-\nassociated", "C05_ALIGNED_LABELS.tsv.gz", "positive", "#C65A46", 437),
]


def mm(v: float) -> float:
    return v * 72 / 25.4


def rect_mm(x: float, y: float, w: float, h: float) -> fitz.Rect:
    return fitz.Rect(mm(x), mm(y), mm(x + w), mm(y + h))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def export_panel_a_from_original_ppt() -> tuple[Path, Path]:
    """Preserve the original editable panel-a geometry and remove only its title.

    The copied PPTX remains editable. PowerPoint exports its shapes and text to a
    vector PDF, which is then placed directly into the composite figure.
    """
    if not CLINICAL_PPT.is_file():
        raise FileNotFoundError(CLINICAL_PPT)
    editable = OUT / "Fig4a_CLINICAL_CONTRASTS_EDITABLE_no_title.pptx"
    target = OUT / "Fig4a_clinical_contrasts_from_original_ppt.pdf"
    shutil.copy2(CLINICAL_PPT, editable)
    from pptx import Presentation
    presentation = Presentation(editable)
    removed_title = 0
    removed_panel_letter = 0
    for slide in presentation.slides:
        for shape in list(slide.shapes):
            if not getattr(shape, "has_text_frame", False):
                continue
            text = shape.text.strip()
            if text == "Clinical contrasts":
                shape._element.getparent().remove(shape._element)
                removed_title += 1
            elif text == "a":
                # The composite adds a single standardized a-f panel-letter layer.
                shape._element.getparent().remove(shape._element)
                removed_panel_letter += 1
    if removed_title != 1 or removed_panel_letter != 1:
        raise RuntimeError(
            "Expected exactly one 'Clinical contrasts' title and one original panel letter; "
            f"removed title={removed_title}, letter={removed_panel_letter}"
        )
    presentation.save(editable)
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$source='{editable}'; $target='{target}'; "
        "$app=New-Object -ComObject PowerPoint.Application; "
        "try { $deck=$app.Presentations.Open($source,$true,$false,$false); "
        "$deck.SaveAs($target,32); $deck.Close() } "
        "finally { $app.Quit(); "
        "[void][Runtime.InteropServices.Marshal]::ReleaseComObject($app); "
        "[GC]::Collect(); [GC]::WaitForPendingFinalizers() }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
    if not target.is_file():
        raise FileNotFoundError(target)
    with fitz.open(target) as doc:
        text = doc[0].get_text()
        if "Clinical contrasts" in text or re.search(r"(?m)^a$", text):
            raise RuntimeError("Panel-a title or original panel letter was not removed from the exported PDF")
        if not all(code in text for code in ("C01", "C02", "C03", "C04", "C05")):
            raise RuntimeError("Original panel-a contrast labels are incomplete after export")
    return target, editable


def redraw_bottom() -> None:
    """Replot the four pathway columns from frozen source tables, as editable text."""
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("draw_bottom_from_frozen_tables.py")),
        "--input-dir", str(FROZEN_BOTTOM),
        "--module-dir", str(FROZEN_BOTTOM),
        "--output-dir", str(OUT / "bottom_redraw"),
    ], check=True)
    if not BOTTOM.is_file():
        raise FileNotFoundError(BOTTOM)


def draw_umaps() -> tuple[Path, list[Path]]:
    umap_file = SRC / "FROZEN_HVG3000_UMAP.tsv.gz"
    umap = pd.read_csv(umap_file, sep="\t")
    xcol, ycol = ("UMAP_1", "UMAP_2") if "UMAP_1" in umap.columns else ("UMAP1", "UMAP2")
    ids = umap["cell_id"].astype(str)
    xy = umap[[xcol, ycol]].to_numpy(dtype=np.float32)
    extent = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    fig, axs = plt.subplots(2, 3, figsize=(95 / 25.4, 92 / 25.4))
    label_files = []
    umap_codes = ["C01", "C02", "C03", "C04", "C05−", "C05+"]
    for ax, (title, name, polarity, color, expected), code in zip(axs.flat, UMAP_SPECS, umap_codes):
        path = SRC / name
        label_files.append(path)
        labels = pd.read_csv(path, sep="\t")
        valcol = next(c for c in labels.columns if c != "cell_id")
        vals = labels[valcol].astype(str).str.strip().str.lower()
        if polarity == "positive":
            mask = vals.isin(["1", "scpas+", "positive", "pos", "symptomatic_associated_target"])
        else:
            mask = vals.isin(["-1", "scpas-", "negative", "neg", "asymptomatic_associated_reference"])
        selected_ids = set(labels.loc[mask, "cell_id"].astype(str))
        highlighted = ids.isin(selected_ids).to_numpy()
        if int(highlighted.sum()) != expected:
            raise ValueError(f"Frozen UMAP count mismatch for {name}: {highlighted.sum()} vs {expected}")
        ax.scatter(xy[:, 0], xy[:, 1], s=0.18, c="#D7D7D7", alpha=0.48, linewidths=0, rasterized=True)
        ax.scatter(xy[highlighted, 0], xy[highlighted, 1], s=0.75, c=color, alpha=0.94, linewidths=0, rasterized=True)
        ax.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{code}  {title}\n(n={expected:,})", fontsize=5.6, pad=1.5, linespacing=1.02)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D7D7D7", markeredgecolor="none", markersize=2.8, label="Background cells"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="none", markersize=2.8, label="Phenotype-associated cells"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=5.1, bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.075, wspace=.025, hspace=.17)
    target = OUT / "Fig4b_UMAP_without_c1_c6.pdf"
    fig.savefig(target, dpi=450)
    fig.savefig(OUT / "Fig4b_UMAP_without_c1_c6.svg", dpi=450)
    plt.close(fig)
    return target, [umap_file, *label_files]


def draw_cd() -> Path:
    comp_path = OLD / "audits" / "C05_CELL_COMPOSITION_AUDIT.tsv"
    enrich_path = SRC / "Fig4d_lineage_enrichment_source_used.tsv"
    comp = pd.read_csv(comp_path, sep="\t")
    enr = pd.read_csv(enrich_path, sep="\t")
    if set(comp.group) != set(GROUPS) or set(enr.group) != set(GROUPS):
        raise ValueError("Composition/enrichment group mismatch")
    fig = plt.figure(figsize=(86 / 25.4, 93 / 25.4))

    # c: the six frozen associated-cell composition vectors.
    ax = fig.add_axes([.13, .655, .855, .275])
    x = np.arange(6)
    bottom = np.zeros(6)
    for lineage in LINEAGES:
        sub = comp.set_index(["group", "lineage"])
        vals = np.array([float(sub.loc[(group, lineage), "fraction"]) * 100 for group in GROUPS])
        ax.bar(x, vals, bottom=bottom, width=.68, color=COLORS[lineage], edgecolor="white", linewidth=.25)
        for i, (v, base) in enumerate(zip(vals, bottom)):
            if v >= 10:
                ax.text(i, base + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=5.1, color="white")
        bottom += vals
    totals = comp.groupby("group")["total"].first()
    for i, group in enumerate(GROUPS):
        ax.text(i, 102.5, f"n={int(totals[group]):,}", ha="center", va="bottom", fontsize=5.3)
    ax.set_ylim(0, 111)
    ax.set_ylabel("Associated cells (%)", fontsize=5.6, labelpad=1)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x, GROUP_DISPLAY)
    ax.tick_params(axis="x", labelsize=5.0, pad=1, rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.tick_params(axis="y", labelsize=5.2, pad=1, length=2)
    ax.grid(axis="y", color="#DADADA", linewidth=.35)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=COLORS[p], label=p) for p in LINEAGES], loc="upper center", bbox_to_anchor=(.55, .47), ncol=5, fontsize=5.0, frameon=False, handlelength=.8, columnspacing=.6, handletextpad=.25)
    # panel letter c is added at the composite level for uniform alignment.

    # d: vector rectangles and frozen log2 odds/FDR values.
    dax = fig.add_axes([.30, .075, .61, .34])
    cmap = LinearSegmentedColormap.from_list("r4", ["#6C8199", "#F4EFE5", "#A45A4A"])
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    data = enr.set_index(["group", "lineage"])
    for row, group in enumerate(GROUPS):
        for col, pop in enumerate(LINEAGES):
            item = data.loc[(group, pop)]
            val = float(item.log2_OR)
            q = float(item.BH_FDR)
            dax.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=cmap(norm(np.clip(val, -5, 5))), edgecolor="white", linewidth=.6))
            stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
            if stars:
                dax.text(col, row, stars, ha="center", va="center", fontsize=5.3, fontweight="bold", color="#262626")
    dax.set_xlim(-.5, 4.5)
    dax.set_ylim(5.5, -.5)
    dax.set_xticks(range(5), LINEAGES)
    dax.set_yticks(range(6), GROUP_DISPLAY)
    dax.tick_params(length=0, pad=1.5, labelsize=5.0)
    for tick in dax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_rotation_mode("anchor")
        tick.set_ha("right")
    for spine in dax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.935, .075, .018, .34])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([-5, 0, 5])
    cb.ax.tick_params(labelsize=5.0, length=1, pad=1)
    cb.set_label("log2 odds ratio", fontsize=5.0, labelpad=1)
    # panel letter d is added at the composite level for uniform alignment.
    target = OUT / "Fig4cd_composition_and_enrichment.pdf"
    fig.savefig(target)
    fig.savefig(OUT / "Fig4cd_composition_and_enrichment.svg")
    plt.close(fig)
    return target


def assemble(a_pdf: Path, b_pdf: Path, cd_pdf: Path) -> Path:
    # Panel a now receives its own complete 64-mm row so the original editable
    # PPT schematic, including the lower legend, is not reduced or clipped.
    width, height = 190, 246
    doc = fitz.open()
    page = doc.new_page(width=mm(width), height=mm(height))
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
    with fitz.open(a_pdf) as source:
        # The original PPT canvas includes a title band. The title is removed at
        # source, so crop only that now-empty whitespace while preserving every
        # clinical-state box, contrast arrow, count, and legend as vector art.
        page.show_pdf_page(
            rect_mm(4, 3, 182, 64),
            source,
            0,
            keep_proportion=True,
            clip=fitz.Rect(40, 50, 1340, 510),
        )
    with fitz.open(b_pdf) as source:
        page.show_pdf_page(rect_mm(4, 69, 95, 93), source, 0, keep_proportion=True)
    with fitz.open(cd_pdf) as source:
        page.show_pdf_page(rect_mm(100, 69, 86, 93), source, 0, keep_proportion=True)
    with fitz.open(BOTTOM) as source:
        page.show_pdf_page(rect_mm(3, 170, 184, 72.4), source, 0, keep_proportion=True)
    for label, x, y in [("a", 2.6, 8.0), ("b", 2.6, 71.4), ("c", 98.6, 71.4), ("d", 98.6, 122.0), ("e", 2.6, 174.0), ("f", 98.6, 174.0)]:
        page.insert_text((mm(x), mm(y)), label, fontsize=8.1, fontname="hebo", color=(.15, .19, .22), overlay=True)
    target = OUT / "Figure4_RESULT4_FINAL_AUDITED.pdf"
    doc.save(target, garbage=4, deflate=True)
    doc.close()
    return target


def clean_svg_display_text(svg_content: str) -> str:
    """Remove stale expanded text labels from embedded legacy vector objects."""
    svg_content = re.sub(
        r'(<text[^>]*>\s*<tspan[^>]*>)M\. tuberculosis(</tspan></text>)\s*'
        r'<text[^>]*>\s*<tspan[^>]*>infection</tspan></text>',
        r'\1TB infection\2',
        svg_content,
        count=1,
        flags=re.S,
    )
    svg_content = svg_content.replace("M. tuberculosis infection", "TB infection")
    return svg_content


def main() -> None:
    redraw_bottom()
    a_pdf, a_editable = export_panel_a_from_original_ppt()
    b_pdf, umap_sources = draw_umaps()
    cd_pdf = draw_cd()
    final = assemble(a_pdf, b_pdf, cd_pdf)
    with fitz.open(final) as doc:
        doc[0].get_pixmap(dpi=300, alpha=False).save(OUT / "Figure4_RESULT4_FINAL_AUDITED.png")
        svg_file = OUT / "Figure4_RESULT4_FINAL_AUDITED.svg"
        svg_content = clean_svg_display_text(doc[0].get_svg_image(text_as_path=False))
        svg_file.write_text(svg_content, encoding="utf-8")
        if "<text" not in svg_content:
            raise RuntimeError("Composite SVG text was converted to paths")
        if any(stale in svg_content for stale in ("TB-negative", "M. tuberculosis", "State-associated cells")):
            raise RuntimeError("Composite SVG still contains stale display labels")
        words = [w[4].lower() for w in doc[0].get_text("words")]
        # The requested contrast codes replace the old lower-case c1-c6 tags.
        assert not any(w in {f"c{i}" for i in range(1, 7)} for w in words)
        assert all(code in doc[0].get_text() for code in ("C01", "C02", "C03", "C04"))
        qa = {
            "page_count": len(doc),
            "size_mm": [190, 246],
            "pdf_bytes": final.stat().st_size,
            "embedded_images": len(doc[0].get_images(full=True)),
            "fonts": sorted({font[3] for font in doc[0].get_fonts()}),
            "umap_codes": ["C01", "C02", "C03", "C04", "C05−", "C05+"],
            "umap_labels_c1_to_c6_removed": True,
            "bottom_replotted_from_frozen_tables": True,
            "svg_text_elements": svg_content.count("<text"),
            "scientific_recomputation": False,
            "panel_a_source": "original_editable_ppt_without_clinical_contrasts_title",
            "panel_a_ppt_preserved_as_editable_copy": True,
            "umap_point_layers_rasterized_at_dpi": 450,
        }
    bottom_sources = [FROZEN_BOTTOM / name for name in (
        "Fig4_asymptomatic_GO_KEGG_source_data.tsv",
        "Fig4_symptomatic_GO_KEGG_source_data.tsv",
        "Fig4_asymptomatic_metabolic_source_data.tsv",
        "Fig4_symptomatic_metabolic_source_data.tsv",
        "ASYMPTOMATIC_METABOLIC_MODULES.tsv",
        "SYMPTOMATIC_METABOLIC_MODULES.tsv",
    )]
    sources = [CLINICAL_PPT, a_editable, a_pdf, *bottom_sources, OLD / "audits" / "C05_CELL_COMPOSITION_AUDIT.tsv", SRC / "Fig4d_lineage_enrichment_source_used.tsv", *umap_sources]
    with (OUT / "PANEL_SOURCE_MANIFEST.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source\tsha256\trole\n")
        for path in sources:
            role = "a" if path in {CLINICAL_PPT, a_editable, a_pdf} else "e_f" if path in bottom_sources else "b" if path in umap_sources else "c_d"
            handle.write(f"{path}\t{digest(path)}\t{role}\n")
    (OUT / "VECTOR_QC.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    manuscript_pdf = MANUSCRIPT / "Figure 4.pdf"
    manuscript_svg = MANUSCRIPT / "Figure 4.svg"
    manuscript_png = MANUSCRIPT / "Figure 4.png"
    shutil.copy2(final, manuscript_pdf)
    shutil.copy2(svg_file, manuscript_svg)
    shutil.copy2(OUT / "Figure4_RESULT4_FINAL_AUDITED.png", manuscript_png)
    print(json.dumps({"pdf": str(final), "svg": str(svg_file), "preview": str(OUT / "Figure4_RESULT4_FINAL_AUDITED.png"), "manuscript_pdf": str(manuscript_pdf), "manuscript_svg": str(manuscript_svg), "manuscript_png": str(manuscript_png), "bottom_editable_pdf": str(BOTTOM), **qa}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
