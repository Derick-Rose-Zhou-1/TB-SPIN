#!/usr/bin/env python3
"""Result5 Pcut sensitivity from the frozen Result3 continuous scores only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

RUN = Path('inputs/TB_STATE_RESULT3_V126_ENDPOINT_SPECIFIC_PROJECT/TB_STATE_RESULT3_V126_ENDPOINT_SPECIFIC/run_20260815_193253')
SOURCE = RUN / '20_FINAL_PCUT_DEG_SIGNATURE_AUC_SEARCH/01_FROZEN_SCORES/FROZEN_SCORES.parquet'
OUT = Path('inputs/TB_STATE_RESULT5_MAX10GPU_FINAL_20260901/61_pcut_sensitivity')
PCUTS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99]
ENDPOINTS = {'H2_180D': 'U6', 'H3_365D': 'U12', 'H1_COX': 'UCOX'}


def lineage_vector(frame: pd.DataFrame, mask: np.ndarray, levels: list[str]) -> np.ndarray:
    count = frame.loc[mask, 'lineage'].astype(str).value_counts()
    values = np.asarray([count.get(level, 0.0) for level in levels], dtype=float)
    return (values + 1e-12) / (values.sum() + 1e-12 * len(values))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    ref = frame.loc[frame['disease_group'].astype(str).eq('LTBI')].copy()
    if ref.empty:
        raise RuntimeError('LTBI reference cells unavailable in frozen score table')
    levels = sorted(frame['lineage'].astype(str).unique())
    rows, jaccard_rows, overlap_rows, jsd_rows = [], [], [], []
    for endpoint, column in ENDPOINTS.items():
        values = frame[column].to_numpy(float)
        ref_values = ref[column].to_numpy(float)
        threshold95 = float(np.quantile(ref_values, 0.95, method='higher'))
        mask95 = values >= threshold95
        denominator95 = max(int(mask95.sum()), 1)
        for pcut in PCUTS:
            threshold = float(np.quantile(ref_values, pcut, method='higher'))
            mask = values >= threshold
            shared = int(np.logical_and(mask, mask95).sum())
            union = int(np.logical_or(mask, mask95).sum())
            minimum = max(min(int(mask.sum()), int(mask95.sum())), 1)
            jaccard = shared / union if union else float('nan')
            overlap = shared / minimum
            jsd = float(jensenshannon(lineage_vector(frame, mask, levels), lineage_vector(frame, mask95, levels), base=2.0) ** 2)
            base = {
                'endpoint': endpoint, 'score_column': column, 'Pcut': pcut,
                'reference_n': len(ref), 'threshold': threshold,
                'n_high': int(mask.sum()), 'fraction_high': float(mask.mean()),
                'p95_n_high': denominator95, 'shared_with_P95': shared,
            }
            rows.append(base)
            jaccard_rows.append({**base, 'jaccard_vs_P95': jaccard})
            overlap_rows.append({**base, 'overlap_coefficient_vs_P95': overlap})
            jsd_rows.append({**base, 'lineage_JSD_vs_P95': jsd})
    pd.DataFrame(rows).to_csv(OUT / 'PCUT_CELL_COUNTS.tsv', sep='\t', index=False)
    pd.DataFrame(jaccard_rows).to_csv(OUT / 'PCUT_JACCARD.tsv', sep='\t', index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / 'PCUT_OVERLAP.tsv', sep='\t', index=False)
    pd.DataFrame(jsd_rows).to_csv(OUT / 'PCUT_LINEAGE_JSD.tsv', sep='\t', index=False)
    master = pd.DataFrame(rows).merge(pd.DataFrame(jaccard_rows)[['endpoint', 'Pcut', 'jaccard_vs_P95']], on=['endpoint', 'Pcut']).merge(pd.DataFrame(overlap_rows)[['endpoint', 'Pcut', 'overlap_coefficient_vs_P95']], on=['endpoint', 'Pcut']).merge(pd.DataFrame(jsd_rows)[['endpoint', 'Pcut', 'lineage_JSD_vs_P95']], on=['endpoint', 'Pcut'])
    master.to_csv(OUT / 'PCUT_MASTER.tsv', sep='\t', index=False)
    quick = master.loc[master['Pcut'].isin([0.90, 0.925, 0.95, 0.975, 0.99])]
    (OUT / 'QUICK_5_3_STATUS.json').write_text(json.dumps({
        'status': 'CORE_STAGE1_COMPLETE', 'source': str(SOURCE), 'formal_cells': int(len(frame)),
        'reference_cells': int(len(ref)), 'thresholds': PCUTS, 'quick_rows': int(len(quick)),
        'retrained_model': False, 'recomputed_scores': False,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
