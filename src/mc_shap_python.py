import numpy as np
from typing import Callable, Optional

def _baseline_vector(X_bg: np.ndarray, method: str = "mean"):
    return X_bg.mean(axis=0) if method == "mean" else np.median(X_bg, axis=0)

def mc_shap_single(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, X_bg: np.ndarray, P: int = 128, seed: Optional[int] = 0):
    rng = np.random.default_rng(seed)
    D = x.shape[0]
    phi = np.zeros(D, dtype=np.float64)
    x0 = _baseline_vector(X_bg, method="mean")
    a = x0.copy()
    prev = f(a[None, :])[0]
    for p in range(P):
        perm = rng.permutation(D)
        a[:] = x0
        prev = f(a[None, :])[0]
        for feat in perm:
            a[feat] = x[feat]
            cur = f(a[None, :])[0]
            phi[feat] += (cur - prev)
            prev = cur
    return phi / P

def mc_shap_batch(f, X, X_bg, P=128, seed=0):
    out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = mc_shap_single(f, X[i], X_bg, P=P, seed=seed+i)
    return out
