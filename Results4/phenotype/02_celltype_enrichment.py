from __future__ import annotations

import argparse
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from r4_common import CONTRASTS, MAJOR_CELLTYPES, bh_adjust, load_config, odds_ratio_ci, read_scores, safe_to_csv, score_path, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PAS+ versus neutral major-cell enrichment.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["out"])
    log = setup_logging(out / "logs/02_celltype_enrichment.log", "celltype_enrichment")
    atlas = ad.read_h5ad(cfg["atlas"], backed="r")
    meta = atlas.obs[["celltype", "celltype_fine", "sample_id_std", "dataset"]].copy()
    meta.index = atlas.obs_names.astype(str)
    atlas.file.close()

    enrichment_rows: list[dict] = []
    composition_rows: list[dict] = []
    for contrast, (reference, target, _, _, _) in CONTRASTS.items():
        score = read_scores(score_path(cfg, contrast)).set_index("cell_id")
        score = score.loc[meta.index]
        labels = score["scPAS_label"].astype(str)
        label_counts = labels.value_counts()
        pos_total = int(label_counts.get("scPAS+", 0))
        neutral_total = int(label_counts.get("0", 0))
        for label in ["scPAS+", "scPAS-", "0"]:
            mask = labels.eq(label)
            for ct in MAJOR_CELLTYPES:
                n = int((mask & meta["celltype"].eq(ct)).sum())
                composition_rows.append({
                    "contrast": contrast, "reference_state": reference, "target_state": target,
                    "association_label": label, "celltype": ct, "cell_n": n,
                    "label_total_n": int(mask.sum()), "composition_fraction": n / int(mask.sum()) if mask.sum() else np.nan,
                })
        pvals: list[float] = []
        start = len(enrichment_rows)
        for ct in MAJOR_CELLTYPES:
            is_ct = meta["celltype"].eq(ct)
            is_pos = labels.eq("scPAS+")
            is_neutral = labels.eq("0")
            a = int((is_pos & is_ct).sum())
            b = int((is_pos & ~is_ct).sum())
            c = int((is_neutral & is_ct).sum())
            d = int((is_neutral & ~is_ct).sum())
            odds, lo, hi = odds_ratio_ci(a, b, c, d)
            p = float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)
            pvals.append(p)
            enrichment_rows.append({
                "contrast": contrast, "reference_state": reference, "target_state": target,
                "total_cell_n": len(meta), "PASplus_n": pos_total, "PASplus_fraction": pos_total / len(meta),
                "PASminus_n": int(label_counts.get("scPAS-", 0)),
                "PASminus_fraction": int(label_counts.get("scPAS-", 0)) / len(meta),
                "neutral_n": neutral_total, "celltype": ct, "PASplus_celltype_n": a,
                "PASplus_composition_fraction": a / pos_total if pos_total else np.nan,
                "total_celltype_n": int(is_ct.sum()),
                "within_celltype_PASplus_fraction": a / int(is_ct.sum()) if is_ct.sum() else np.nan,
                "OR": odds, "CI_low": lo, "CI_high": hi, "P": p,
                "background_definition": "PASplus_vs_Neutral; PASminus excluded",
            })
        q = bh_adjust(pvals)
        for row, fdr in zip(enrichment_rows[start:], q):
            row["FDR"] = fdr
        log.info("%s enrichment completed", contrast)

    safe_to_csv(pd.DataFrame(enrichment_rows), out / "01_CELL_ENRICHMENT/R4_CELLTYPE_ENRICHMENT.tsv")
    safe_to_csv(pd.DataFrame(composition_rows), out / "01_CELL_ENRICHMENT/R4_SCPAS_COMPOSITION.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
