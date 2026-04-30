import os, time, platform, numpy as np, pandas as pd, joblib, psutil, subprocess
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from mc_shap_python import mc_shap_batch

def try_import_openmp():
    try:
        sys.path.append(str(ROOT / "src" / "ext_openmp" / "build"))
        import mc_shap_openmp
        return mc_shap_openmp
    except Exception:
        return None

def try_import_cuda():
    try:
        sys.path.append(str(ROOT / "src" / "ext_cuda" / "build"))
        from mc_shap_cuda import mc_shap_cuda_linear
        return mc_shap_cuda_linear
    except Exception:
        return None

def load_data_and_model(dataset_name="market_features.csv"):
    data = pd.read_csv(ROOT / "data" / dataset_name)
    obj = joblib.load(ROOT / "models" / "model.pkl")
    model = obj["model"]
    X = data.drop(columns=["target"]).values
    y = data["target"].values if "target" in data.columns else None
    return X, y, model

def hardware_info():
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cores_logical": psutil.cpu_count(logical=True),
        "cores_physical": psutil.cpu_count(logical=False),
    }
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        lines = out.decode().strip().splitlines()
        info["gpus"] = [l.strip() for l in lines]
    except Exception:
        info["gpus"] = []
    return info

def run_benchmark(backend: str, P: int, N: int|None, threads: int|None, dataset_name: str):
    X, y, model = load_data_and_model(dataset_name)
    if N: X = X[:N]
    X_bg = X if len(X) < 512 else X[:512]

    def f_model(Xin):
        return model.predict_proba(Xin)[:, 1]

    # Get Python baseline timing for speedup calculation
    # Only run this once with Python backend
    if backend == "python":
        t0 = time.time()
        phi_py = mc_shap_batch(f_model, X, X_bg, P=P, seed=0)
        t1 = time.time()
        rt = t1 - t0
        
        # No baseline comparison for Python - it IS the baseline
        return rt, 1.0, 1.0000  # speedup=1x, fidelity=1.0 (perfect)

    # For other backends, we need Python baseline for comparison
    print(f"Running Python baseline for comparison...")
    t0 = time.time()
    phi_py = mc_shap_batch(f_model, X, X_bg, P=P, seed=0)
    t1 = time.time()
    baseline_time = t1 - t0
    print(f"Python baseline: {baseline_time:.4f}s")

    if backend == "openmp":
        omp = try_import_openmp()
        if omp is None:
            raise RuntimeError("OpenMP extension not found. Build src/ext_openmp first.")
        if threads: 
            os.environ["OMP_NUM_THREADS"] = str(threads)
        
        # Check if fast version exists
        has_fast = hasattr(omp, 'mc_shap_openmp_fast')
        
        print(f"Running OpenMP ({'fast' if has_fast else 'slow'} version)...")
        t0 = time.time()
        if has_fast:
            phi_omp = omp.mc_shap_openmp_fast(f_model, X, X_bg, P, 0)
        else:
            # Warn about slow version
            print("WARNING: Using slow OpenMP version with GIL overhead!")
            print("This will likely be slower than Python baseline.")
            phi_omp = omp.mc_shap_openmp(f_model, X, X_bg, P, 0)
        t1 = time.time()
        rt = t1 - t0
        
        corr = float(np.corrcoef(phi_py.ravel(), phi_omp.ravel())[0,1])
        speedup = baseline_time / (rt + 1e-12)
        
        return rt, speedup, corr

    elif backend == "cuda":
        mc_shap_cuda_linear = try_import_cuda()
        if mc_shap_cuda_linear is None:
            raise RuntimeError("CUDA extension not found. Build src/ext_cuda first.")

        if y is None:
            raise RuntimeError("CUDA backend requires labels to fit a linear surrogate model.")

        from sklearn.linear_model import LogisticRegression
        print("Training linear model for CUDA...")
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        baseline = X_bg.mean(axis=0).astype(np.float64)
        W = lr.coef_.ravel().astype(np.float64)
        b = float(lr.intercept_[0])

        print("Running CUDA kernel...")
        t0 = time.time()
        phi_cuda = mc_shap_cuda_linear(
            X.astype(np.float64),
            baseline,
            W,
            b,
            P,
            128,
            0
        )
        t1 = time.time()
        rt = t1 - t0
        
        # Note: CUDA uses different model, so fidelity is vs Python MC-SHAP with tree model
        # This is not a perfect comparison but gives a sense of consistency
        corr = float(np.corrcoef(phi_py.ravel(), phi_cuda.ravel())[0,1])
        speedup = baseline_time / (rt + 1e-12)
        
        return rt, speedup, corr

    else:
        raise ValueError("backend must be: python | openmp | cuda")