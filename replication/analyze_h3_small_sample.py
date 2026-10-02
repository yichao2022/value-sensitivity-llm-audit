#!/usr/bin/env python3
"""Small-sample inference for the H3 frame regressions.

Model-specific H3 regressions have only 27 profile clusters each, and several
model-level results now carry interpretive weight (Qwen autonomy reversal,
Mistral equity/access at the margin). This script reports, for every H3
contrast:

  1. asymptotic cluster-robust inference (CR1, no small-sample correction)
  2. CR1 with the standard finite-sample correction (G/(G-1) * (N-1)/(N-K))
  3. wild cluster bootstrap-t p-value (Rademacher weights, null imposed,
     clusterthe same profile clusters), B = 1999

Single source: outputs/frame_delta_analysis.csv (the file behind Figure 3 and
the supplement H3 tables).
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
B = 1999
RNG = np.random.default_rng(20260917)


def design(df):
    X = pd.concat([
        pd.Series(1.0, index=df.index, name="const"),
        pd.get_dummies(df.profile_id, prefix="p", drop_first=True, dtype=float),
        pd.get_dummies(df.frame, prefix="f", drop_first=False, dtype=float).drop(columns=["f_neutral"]),
    ], axis=1)
    return X


def cluster_stats(X, y, groups, k):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g, idx in groups.items():
        Xg = X.iloc[idx].values
        ug = u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(groups); N = len(y)
    V_plain = XtX_inv @ meat @ XtX_inv
    c = (G / (G - 1)) * ((N - 1) / (N - X.shape[1]))
    V_corr = V_plain * c
    return b, V_plain, V_corr, k


def wcb_p(X, y, groups, col, t_obs, B=B):
    """Wild cluster bootstrap-t with the null imposed on `col`."""
    keep = [c for c in X.columns if c != col]
    Xr, Xu = X[keep].values, X.values
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
    y_hat_r = Xr @ br
    e_r = y - y_hat_r
    glist = list(groups.values())
    XtX_inv = np.linalg.pinv(Xu.T @ Xu)
    j = list(X.columns).index(col)
    count = 0
    for _ in range(B):
        w = RNG.choice([-1.0, 1.0], size=len(glist))
        y_star = y_hat_r.copy()
        for wg, idx in zip(w, glist):
            y_star[idx] = y_hat_r[idx] + wg * e_r[idx]
        b_star = XtX_inv @ (Xu.T @ y_star)
        u_star = y_star - Xu @ b_star
        meat = np.zeros((Xu.shape[1], Xu.shape[1]))
        for idx in glist:
            s = Xu[idx].T @ u_star[idx]
            meat += np.outer(s, s)
        V = XtX_inv @ meat @ XtX_inv
        se = np.sqrt(V[j, j])
        if se > 0 and abs(b_star[j] / se) >= abs(t_obs):
            count += 1
    return (count + 1) / (B + 1)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "frame_delta_analysis.csv").dropna(subset=["delta"])
    contrasts = ["f_autonomy", "f_collective", "f_equity"]
    rows = []
    for model in sorted(df.model.unique()):
        d = df[df.model == model].reset_index(drop=True)
        X = design(d)
        y = d.delta.values.astype(float)
        groups = {g: np.array(idx) for g, idx in d.groupby("profile_id").groups.items()}
        for c in contrasts:
            b, Vp, Vc, _ = cluster_stats(X, y, groups, c)
            j = list(X.columns).index(c)
            t_plain = b[j] / np.sqrt(Vp[j, j])
            t_corr = b[j] / np.sqrt(Vc[j, j])
            from scipy import stats
            p_plain = 2 * stats.t.sf(abs(t_plain), len(groups) - 1)
            p_corr = 2 * stats.t.sf(abs(t_corr), len(groups) - 1)
            p_boot = wcb_p(X.assign(**{c: X[c]}), y, groups, c, t_plain)
            rows.append(dict(model=model, contrast=c.replace("f_", ""), b=b[j],
                             p_CR1=p_plain, p_CR1_corrected=p_corr, p_wild_bootstrap=p_boot))
            print(f"{model:<24} {c.replace('f_',''):<11} b={b[j]:+7.3f}  "
                  f"p_CR1={p_plain:.5f}  p_CR1corr={p_corr:.5f}  p_boot={p_boot:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "h3_small_sample_inference.csv", index=False)

    # pooled specification for reference
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   pd.get_dummies(df.model, prefix="m", drop_first=True, dtype=float),
                   pd.get_dummies(df.profile_id, prefix="p", drop_first=True, dtype=float),
                   pd.get_dummies(df.frame, prefix="f", drop_first=False, dtype=float).drop(columns=["f_neutral"])], axis=1)
    y = df.delta.values.astype(float)
    groups = {g: np.array(idx) for g, idx in df.groupby(["model", "profile_id"]).groups.items()}
    print("\npooled (108 clusters):")
    for c in contrasts:
        b, Vp, Vc, _ = cluster_stats(X, y, groups, c)
        j = list(X.columns).index(c)
        t = b[j] / np.sqrt(Vp[j, j])
        from scipy import stats
        print(f"  {c.replace('f_',''):<11} b={b[j]:+7.3f}  p_CR1={2*stats.t.sf(abs(t), len(groups)-1):.5f}  CI=[{b[j]-1.96*np.sqrt(Vp[j,j]):+.2f}, {b[j]+1.96*np.sqrt(Vp[j,j]):+.2f}]")
    print(f"\nwrote {OUT/'h3_small_sample_inference.csv'}")
