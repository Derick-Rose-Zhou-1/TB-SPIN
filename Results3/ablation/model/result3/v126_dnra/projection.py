"""Frozen formal-cell latent projection utilities."""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def project_latent(model, counts: csr_matrix, batch_size: int, device: str | torch.device) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[0] == 0 or batch_size <= 0:
        raise ValueError("INVALID_PROJECTION_INPUT")
    target = torch.device(device)
    result: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, counts.shape[0], batch_size):
            batch = torch.as_tensor(counts[start : start + batch_size].toarray(), dtype=torch.float32, device=target)
            mean, _ = model.encode(batch)
            if not torch.isfinite(mean).all():
                raise RuntimeError("NONFINITE_FORMAL_LATENT")
            result.append(mean.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(result)
