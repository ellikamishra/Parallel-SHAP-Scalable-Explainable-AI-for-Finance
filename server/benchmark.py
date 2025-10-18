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

    t0 = time.time()
    phi_py = mc_shap_batch(f_model, X, X_bg, P=P, seed=0)
    t1 = time.time()
    base_time = t1 - t0

    if backend == "python":
        return base_time, 1.0, 1.0

    elif backend == "openmp":
        omp = try_import_openmp()
        if omp is None:
            raise RuntimeError("OpenMP extension not found. Build src/ext_openmp first.")
        if threads: os.environ["OMP_NUM_THREADS"] = str(threads)
        t0 = time.time()
        phi_omp = omp.mc_shap_openmp(f_model, X, X_bg, P, 0)
        t1 = time.time()
        corr = float(np.corrcoef(phi_py.ravel(), phi_omp.ravel())[0,1])
        return (t1 - t0), base_time / (t1 - t0 + 1e-12), corr

    elif backend == "cuda":
        raise NotImplementedError("CUDA fastpath not wired in server benchmark.")

    else:
        raise ValueError("backend must be: python | openmp | cuda")
