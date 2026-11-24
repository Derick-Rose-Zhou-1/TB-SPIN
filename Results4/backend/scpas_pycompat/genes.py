from __future__ import annotations

def common_genes(bulk_genes, single_cell_genes):
    """R base::intersect semantics: unique first-vector values in first-vector order."""
    sc = set(single_cell_genes)
    seen = set()
    out = []
    for g in bulk_genes:
        if g in sc and g not in seen:
            seen.add(g); out.append(g)
    return out
