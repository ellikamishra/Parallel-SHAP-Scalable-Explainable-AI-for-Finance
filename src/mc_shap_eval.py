import time, joblib, numpy as np, pandas as pd, sys
from pathlib import Path
from importlib import import_module
from mc_shap_python import mc_shap_batch

MODEL = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
DATA  = Path(__file__).resolve().parents[1] / "data" / "market_features.csv"

def ensure_model():
    try:
        obj = joblib.load(MODEL)
    except FileNotFoundError:
        print("Model artifact missing, training a fresh model via train_model.py")
        from train_model import main as train_model_main
        train_model_main()
        obj = joblib.load(MODEL)
    except ValueError as e:
        if "node array from the pickle has an incompatible dtype" in str(e):
            print("Existing model pickle incompatible with current scikit-learn; retraining.")
            from train_model import main as train_model_main
            train_model_main()
            obj = joblib.load(MODEL)
        else:
            raise
    return obj

def main():
    obj = ensure_model()
    model = obj['model']
    df = pd.read_csv(DATA)
    X = df.drop(columns=['target']).values
    y = df['target'].values if 'target' in df.columns else None
    X_bg = X if len(X)<512 else X[:512]
    X_test = X

    def f_model(Xin):
        return model.predict_proba(Xin)[:, 1]

    t0 = time.time()
    phi_py = mc_shap_batch(f_model, X_test, X_bg, P=64, seed=0)
    t1 = time.time()
    print(f"Python MC-SHAP: {(t1-t0):.4f}s, shape={phi_py.shape}")

    try:
        sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "ext_openmp" / "build"))
        omp = import_module("mc_shap_openmp")
        t0 = time.time()
        phi_omp = omp.mc_shap_openmp(f_model, X_test, X_bg, 64, 0)
        t1 = time.time()
        print(f"OpenMP MC-SHAP: {(t1-t0):.4f}s, shape={phi_omp.shape}")
        corr = float(np.corrcoef(phi_py.ravel(), phi_omp.ravel())[0,1])
        print(f"Fidelity corr (py vs omp): {corr:.4f}")
    except Exception as e:
        print("OpenMP extension not available yet:", e)

    try:
        sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "ext_cuda" / "build"))
        from mc_shap_cuda import mc_shap_cuda_linear
        baseline = X_bg.mean(axis=0)
        if y is not None:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=1000).fit(X, y)
            W = lr.coef_.ravel().astype(np.float64)
            b = float(lr.intercept_[0])
            phi_cuda = mc_shap_cuda_linear(X_test.astype(np.float64), baseline.astype(np.float64), W, b, 64, 128, 0)
            print("CUDA linear MC-SHAP shape:", phi_cuda.shape)
    except Exception as e:
        print("CUDA extension not available yet:", e)

if __name__ == "__main__":
    main()
