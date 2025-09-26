"""Train anomaly detection models on the provided CSV and save artifacts.
Usage: python train.py
"""
from pathlib import Path
try:
    import joblib
except Exception:
    joblib = None
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("Functions") / "financial_anomaly_data.csv"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)
    # Robustly parse Timestamp into datetime; coerce errors
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    # Fill missing timestamps with a constant and extract hour
    df["Hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
    X = df[["Amount", "Hour"]].copy()
    cat_cols = ["TransactionType", "Location", "Merchant"]
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([X, df_cat], axis=1)
    X = X.fillna(0)
    # If dataset is large, sample to speed up training
    max_rows = 50000
    if X.shape[0] > max_rows:
        X = X.sample(n=max_rows, random_state=42).reset_index(drop=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    try:
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    except Exception:
        with open(MODELS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    return Xs, X.columns.tolist()


def train_classical(X):
    models = {}
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(X)
    models["isolation_forest"] = iso

    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.01, novelty=True)
    lof.fit(X)
    models["lof"] = lof

    ocsvm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    ocsvm.fit(X)
    models["one_class_svm"] = ocsvm

    for name, m in models.items():
        try:
            joblib.dump(m, MODELS_DIR / f"{name}.joblib")
        except Exception:
            with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(m, f)




def main():
    Xs, feature_names = load_and_preprocess()
    train_classical(Xs)
    with open(RESULTS_DIR / "features.json", "w") as f:
        json.dump(feature_names, f)


if __name__ == "__main__":
    main()
