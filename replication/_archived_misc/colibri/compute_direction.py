#!/usr/bin/env python3
"""Compute refusal direction via PCA on harmful-residuals (single-class, LLM-LAT style).

Reads residual dumps from a COLLECT run (each [n_layers][D] float32, first-token
prefill) and takes, per layer, the first principal component of the
harmful activations as the refusal direction. Also computes the difference-of-means
direction (harmful - harmless) as a comparison. Writes the diff-of-means direction
(unit-normalized, sign-aligned) for COLI_STEER_FILE.

Usage:
  python3 compute_direction.py --harmful <dir> --harmless <dir> --out directions.bin
  python3 compute_direction.py --harmful <dir> --harmless <dir> --out directions.bin --method pca
"""
import os, sys, struct, argparse, glob
from pathlib import Path
import numpy as np

def load_residuals(d):
    f = Path(d) / "residuals.bin"
    if not f.exists(): return None
    raw = f.read_bytes()
    if len(raw) < 8: return None
    n_layers, D = struct.unpack("<ii", raw[:8])
    arr = np.frombuffer(raw[8:8+n_layers*D*4], dtype=np.float32).reshape(n_layers, D)
    return arr

def load_class(base):
    paths = sorted(glob.glob(os.path.join(base, "p*")))
    arrs = []
    for d in paths:
        a = load_residuals(d)
        if a is not None:
            arrs.append(a)
    if not arrs:
        raise SystemExit(f"no residuals under {base}")
    return np.stack(arrs, axis=0)

def first_pc(X):  # X: [N, D] -> [D], first principal component
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD: rows are samples. Use np.linalg.svd on Xc (N x D).
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[0]  # [D]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful", required=True)
    ap.add_argument("--harmless", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="diff", choices=["diff", "pca"])
    args = ap.parse_args()

    H = load_class(args.harmful)
    G = load_class(args.harmless)
    L, D = H.shape[1], H.shape[2]
    print(f"[dir] harmful={H.shape} harmless={G.shape}")

    if args.method == "pca":
        # Per-layer first PC of the harmful residuals (refusal axis per layer)
        r = np.zeros((L, D), dtype=np.float64)
        for li in range(L):
            r[li] = first_pc(H[:, li, :])
    else:
        r = (H.mean(0) - G.mean(0))  # [L, D]

    # sign-align: pick the sign so that the direction's mean matches the
    # harmful-minus-harmless diff sign (refusal should reduce it)
    mean_sign = np.sign(r.sum(axis=1))
    flip = (mean_sign < 0).astype(np.float64)
    r = np.where(flip[:, None] > 0, -r, r)

    norms = np.linalg.norm(r, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    r_unit = (r / norms).astype(np.float32)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<ii", L, D))
        f.write(r_unit.tobytes())
    mags = np.linalg.norm(r, axis=1)
    print(f"[dir] wrote {args.out}: L={L} D={D} method={args.method}")
    print(f"[dir] |r|: min={mags.min():.4f} med={np.median(mags):.4f} max={mags.max():.4f}")
    print(f"[dir] top5 layers: {np.argsort(mags)[::-1][:5].tolist()}")

if __name__ == "__main__":
    main()
