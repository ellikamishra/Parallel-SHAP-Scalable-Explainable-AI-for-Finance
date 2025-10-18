import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA = Path(__file__).resolve().parents[1] / "data" / "market_features.csv"
OUT  = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

def main():
    df = pd.read_csv(DATA)
    if 'target' not in df.columns:
        raise ValueError("CSV must contain a 'target' column.")
    y = df['target'].values
    X = df.drop(columns=['target']).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
    )
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    print("Train acc:", clf.score(X_train, y_train))
    print("Test acc :", clf.score(X_test, y_test))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': clf, 'feature_names': [c for c in df.columns if c!='target']}, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
