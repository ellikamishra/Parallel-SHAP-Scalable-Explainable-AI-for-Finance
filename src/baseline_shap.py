import time, shap, joblib, pandas as pd
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
DATA  = Path(__file__).resolve().parents[1] / "data" / "market_features.csv"

def main():
    obj = joblib.load(MODEL)
    model = obj['model']
    X = pd.read_csv(DATA).drop(columns=['target']).values
    t0 = time.time()
    explainer = shap.TreeExplainer(model) if hasattr(model, "predict_proba") else shap.Explainer(model)
    _ = explainer(X)
    print(f"Baseline SHAP took {time.time()-t0:.4f}s for {X.shape[0]} samples.")
if __name__ == "__main__":
    main()
