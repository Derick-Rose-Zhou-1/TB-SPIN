"""Participant-level median burden permutation test."""
import hashlib
import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def permutation_p(reference: np.ndarray, target: np.ndarray, seed: int, b: int = 10000) -> float:
    values = np.concatenate([reference, target])
    n_ref = len(reference)
    observed = abs(float(np.median(target) - np.median(reference)))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(b):
        perm = rng.permutation(values)
        statistic = abs(float(np.median(perm[n_ref:]) - np.median(perm[:n_ref])))
        ge += int(statistic >= observed)
    return float((1 + ge) / (b + 1))
