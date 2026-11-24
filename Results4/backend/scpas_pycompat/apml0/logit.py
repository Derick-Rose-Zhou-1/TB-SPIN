"""Reference-compatible port of scPAS v0.2.0 APML0.R/net_logit.R.

Only the binomial family is exposed.  Numerical coordinate-descent work is
delegated to the exact C++ kernels derived from the locked upstream source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import native
from ..r_rng import cv_foldid as r_cv_foldid


@dataclass(frozen=True)
class APML0Result:
    Beta: np.ndarray
    fit: dict[str, np.ndarray]
    penalty: str
    adaptive: np.ndarray | bool
    flag: np.ndarray
    family: str = "binomial"
    Beta0: np.ndarray | None = None
    fit0: dict[str, np.ndarray] | None = None
    lambda_min: float | None = None
    lambda_opt: float | None = None


def _weighted_mean_na(values: np.ndarray, weights: np.ndarray) -> float:
    keep = ~np.isnan(values)
    return float(np.sum(values[keep] * weights[keep]) / np.sum(weights[keep]))


def _first_min(values: np.ndarray) -> int:
    finite = ~np.isnan(values)
    if not np.any(finite):
        raise ValueError("which.min has no non-missing value")
    indices = np.flatnonzero(finite)
    return int(indices[np.argmin(values[finite])])


def _rank_min(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    rank = 1
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = rank
        rank = end + 1
        position = end
    return ranks


def _prepare_network(omega: np.ndarray, sgn: np.ndarray) -> tuple:
    omega1 = np.zeros((omega.shape[0] + 1, omega.shape[1] + 1), dtype=float)
    omega1[1:, 1:] = omega
    np.fill_diagonal(omega1, 0.0)
    omega1 = np.abs(omega1)
    w = native.OmegaC(omega1, np.r_[1, sgn].astype(np.int32))
    # The upstream R wrapper adds one before returning loc to the C++ kernel.
    return w["Omega"], np.asarray(w["loc"], dtype=float) + 1.0, np.asarray(w["nadj"], dtype=np.int32)


def log_l0(
    x: np.ndarray,
    y: np.ndarray,
    Omega: np.ndarray | None = None,
    alpha: float = 1.0,
    lambda_: np.ndarray | None = None,
    nlambda: int = 100,
    rlambda: float | None = None,
    wbeta: np.ndarray | None = None,
    sgn: np.ndarray | None = None,
    nfolds: int = 1,
    foldid: np.ndarray | None = None,
    iL0: bool = True,
    icutB: bool = True,
    ncutB: int = 10,
    ifast: bool = True,
    isd: bool = False,
    keep_beta: bool = False,
    thresh: float = 1e-6,
    maxit: int = 100000,
    threshC: float = 1e-5,
    maxitC: int = 100,
    threshP: float = 1e-5,
    cv_seed: int = 123,
) -> APML0Result:
    x = np.asarray(x, dtype=float, order="F")
    y = np.asarray(y, dtype=float)
    N0, p = x.shape
    wbeta = np.ones(p, dtype=float) if wbeta is None else np.abs(np.asarray(wbeta, dtype=float))
    sgn = np.ones(p, dtype=np.int32) if sgn is None else np.asarray(sgn, dtype=np.int32)

    if np.all(wbeta == 0):
        lambda_ = np.array([0.0])
    if lambda_ is None:
        ilambda = 1
        if rlambda is None:
            rlambda = 0.0001 if N0 > p else 0.01
        lambda_path = float(rlambda) ** (np.arange(nlambda, dtype=float) / (nlambda - 1))
    else:
        ilambda = 0
        lambda_path = np.atleast_1d(np.asarray(lambda_, dtype=float))
        nlambda = len(lambda_path)

    if Omega is None:
        penalty = "Lasso" if alpha == 1 else "Enet"
        adaptive: np.ndarray | bool = bool(np.any(wbeta != 1))
        network = None
    else:
        penalty = "Lasso" if alpha == 1 else "Net"
        adaptive = np.array([np.any(wbeta != 1), np.any(sgn != 1)], dtype=bool)
        network = _prepare_network(np.asarray(Omega, dtype=float), sgn)

    x1 = np.column_stack((np.ones(N0), x))
    p1 = p + 1
    wbeta1 = np.r_[0.0, wbeta]
    wbeta2 = np.where(wbeta1 == 0, 0.0, 1.0)

    if penalty == "Net":
        omega_s, loc, nadj = network
        out = native.NetLogC(x1, y, alpha, lambda_path, nlambda, ilambda,
                             wbeta1, wbeta2, omega_s, loc, nadj, p1, N0,
                             thresh, maxit, threshP)
    else:
        out = native.EnetLogC(x1, y, alpha, lambda_path, nlambda, ilambda,
                              wbeta1, wbeta2, p1, N0, thresh, maxit, threshP)

    nlambdai = int(out["nlambda"])
    if nlambdai == 0:
        raise RuntimeError("upstream APML0 returned zero lambdas")
    lambdai = np.asarray(out["lambda"], dtype=float)[:nlambdai]
    beta = np.asarray(out["Beta"], dtype=float)[:, :nlambdai]
    beta_std = np.asarray(out["BetaSTD"], dtype=float)[:, :nlambdai]
    beta[np.isnan(beta)] = beta_std[np.isnan(beta)]
    flags = np.asarray(out["flag"], dtype=np.int32)[:nlambdai]
    ll = np.asarray(out["LL"], dtype=float)[:nlambdai]
    nzero = np.sum(beta != 0, axis=0)
    pdev = (float(out["ll0"]) - ll) / float(out["ll0"])

    if nfolds == 1 and foldid is None:
        fit = {"lambda": lambdai, "pDev": pdev, "nzero": np.maximum(0, nzero - 1)}
        return APML0Result(beta_std if isd else beta, fit, penalty, adaptive, flags)

    if foldid is None:
        foldid = r_cv_foldid(N0, nfolds, cv_seed)
    else:
        foldid = np.asarray(foldid, dtype=np.int64)
        nfolds = int(np.max(foldid))
    counts = np.array([np.sum(foldid == fold) for fold in range(1, nfolds + 1)], dtype=int)
    train_counts = N0 - counts
    weights = counts.astype(float)

    fold_outputs: list[dict] = []
    cv_rss = np.full((nfolds, nlambdai), np.nan)
    for fold0 in range(nfolds):
        test = foldid == (fold0 + 1)
        y_train = y[~test]
        if np.any(y_train == 0) and np.any(y_train == 1):
            if penalty == "Net":
                fold_out = native.cvNetLogC(
                    x1[~test], y_train, alpha, lambdai, nlambdai,
                    wbeta1, wbeta2, omega_s, loc, nadj, p1,
                    int(train_counts[fold0]), thresh, maxit,
                    x1[test], y[test], int(counts[fold0]), threshP,
                )
            else:
                fold_out = native.cvEnetLogC(
                    x1[~test], y_train, alpha, lambdai, nlambdai,
                    wbeta1, wbeta2, p1, int(train_counts[fold0]),
                    thresh, maxit, x1[test], y[test], int(counts[fold0]), threshP,
                )
            fold_beta = np.asarray(fold_out["Beta"], dtype=float)
            fold_beta_std = np.asarray(fold_out["BetaSTD"], dtype=float)
            fold_beta[np.isnan(fold_beta)] = fold_beta_std[np.isnan(fold_beta)]
            fold_out = dict(fold_out)
            fold_out["Beta"] = fold_beta
            fold_out["BetaSTD"] = fold_beta_std
            n_fold_lambda = int(fold_out["nlambda"])
            cv_rss[fold0, :n_fold_lambda] = -2.0 * np.asarray(fold_out["LLF"][:n_fold_lambda]) * counts[fold0]
        else:
            fold_out = {
                "Beta": np.zeros((p1, nlambdai)),
                "BetaSTD": np.zeros((p1, nlambdai)),
            }
        fold_outputs.append(fold_out)

    cvraw = cv_rss / weights[:, None]
    nfoldi = np.sum(~np.isnan(cvraw), axis=0)
    cvm_main = np.array([_weighted_mean_na(cvraw[:, k], weights) for k in range(nlambdai)])
    cvse = np.array([
        np.sqrt(_weighted_mean_na((cvraw[:, k] - cvm_main[k]) ** 2, weights) / (nfoldi[k] - 1))
        for k in range(nlambdai)
    ])
    indexi = _first_min(cvm_main)
    fit = {
        "lambda": lambdai,
        "pDev": pdev,
        "cvm": cvm_main,
        "cvse": cvse,
        "nzero": np.maximum(nzero - 1, 0),
        "index": np.array(["*" if k == indexi else "" for k in range(nlambdai)]),
    }

    if not iL0:
        selected = beta_std if isd else beta
        return APML0Result(selected if keep_beta else selected[:, indexi], fit,
                           penalty, adaptive, flags, lambda_min=float(lambdai[indexi]))

    if icutB:
        raise NotImplementedError("icutB=TRUE is outside the frozen scPAS APML0 call profile")

    il0 = indexi if ifast else 0
    cvm_l0: list[np.ndarray | None] = [None] * nlambdai
    cv_min = np.full(nlambdai, np.nan)
    while True:
        numi = int(nzero[il0])
        betai = np.column_stack([np.asarray(item["Beta"])[:, il0] for item in fold_outputs])
        beta_stdi = np.column_stack([np.asarray(item["BetaSTD"])[:, il0] for item in fold_outputs])
        betao = np.sum(betai != 0, axis=0)
        numi2 = max(min(int(np.max(betao)), numi), 1)
        cv_l0 = np.full((nfolds, numi2), np.nan)
        for fold0 in range(nfolds):
            test = foldid == (fold0 + 1)
            numj = min(int(betao[fold0]), numi)
            betaj = betai[:, fold0]
            beta_stdj = beta_stdi[:, fold0]
            ranking_values = beta_stdj.copy()
            ranking_values[wbeta1 == 0] = np.max(np.abs(beta_stdj)) + 1
            ranks = _rank_min(-np.abs(ranking_values))
            selected = np.flatnonzero(ranks <= numj)
            selected = selected[np.argsort(ranks[selected], kind="stable")][:numj]
            cv_l0[fold0] = native.cvTrimLogC(
                betaj[selected], numj, numi2, selected.astype(np.int32),
                x1[~test], y[~test], int(train_counts[fold0]),
                x1[test], y[test], int(counts[fold0]), threshC, maxitC, threshP,
            )
        cvraw_l0 = cv_l0 / weights[:, None]
        current = np.array([_weighted_mean_na(cvraw_l0[:, k], weights) for k in range(numi2)])
        cvm_l0[il0] = current
        unpenalized = int(np.sum(wbeta1 == 0))
        cv_min[il0] = np.min(current[unpenalized - 1:])

        il1 = _first_min(cv_min)
        if nlambdai == 1:
            break
        if ifast:
            if np.sum(~np.isnan(cv_min)) > 1:
                next_i, prev_i = min(il1 + 1, nlambdai - 1), max(il1 - 1, 0)
                if not np.isnan(cv_min[next_i]) and not np.isnan(cv_min[prev_i]):
                    break
                if np.isnan(cv_min[next_i]):
                    il0 = il1 + 1
                elif np.isnan(cv_min[prev_i]):
                    il0 = il1 - 1
                else:
                    il0 = _first_min(cv_min)
            elif il0 + 1 < nlambdai:
                il0 += 1
            elif il0 - 1 >= 0:
                il0 -= 1
        else:
            il0 += 1
            if il0 >= nlambdai:
                break

    il0 = _first_min(cv_min)
    beta0 = beta[:, il0].copy()
    beta_std0 = beta_std[:, il0].copy()
    current = cvm_l0[il0]
    unpenalized = int(np.sum(wbeta1 == 0))
    cuti = int(np.argmin(current[unpenalized - 1:]) + unpenalized)
    beta0j = beta_std[:, il0].copy()
    beta0j[wbeta1 == 0] = np.max(np.abs(beta0j)) + 1
    if cuti < len(beta0j):
        cutoff = np.sort(np.abs(beta0j))[::-1][cuti]
        beta0[np.abs(beta0j) <= cutoff] = 0
        beta_std0[np.abs(beta0j) <= cutoff] = 0
    fit0 = {
        "lambda": np.array([lambdai[il0]]),
        "cvm": np.array([cv_min[il0]]),
        "nzero": np.array([cuti - 1]),
    }
    selected = beta_std if isd else beta
    selected0 = beta_std0 if isd else beta0
    return APML0Result(
        selected if keep_beta else selected[:, indexi], fit, penalty, adaptive,
        flags, Beta0=selected0, fit0=fit0,
        lambda_min=float(lambdai[indexi]), lambda_opt=float(lambdai[il0]),
    )


def apml0(
    x: np.ndarray,
    y: np.ndarray,
    family: str = "binomial",
    penalty: str = "Net",
    Omega: np.ndarray | None = None,
    alpha: float = 1.0,
    **kwargs,
) -> APML0Result:
    if family != "binomial":
        raise NotImplementedError("compatibility profile is BINOMIAL_ONLY")
    if penalty == "Net" and Omega is None:
        penalty = "Enet"
    if penalty in {"Enet", "Net"} and alpha == 1.0:
        penalty = "Lasso"
    if alpha != 1.0:
        penalty = "Enet" if Omega is None else "Net"
    else:
        penalty = "Lasso"
    # LogL0 derives the effective penalty again from alpha/Omega exactly as R.
    return log_l0(x, y, Omega=Omega, alpha=alpha, **kwargs)
