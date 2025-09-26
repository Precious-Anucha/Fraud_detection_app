import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
try:
    import joblib
except Exception:
    joblib = None

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
DATA_PATH = Path("Functions") / "financial_anomaly_data.csv"


def load_models():
    models = {}
    for name in ["isolation_forest", "lof", "one_class_svm"]:
        path_joblib = MODELS_DIR / f"{name}.joblib"
        path_pkl = MODELS_DIR / f"{name}.pkl"
        if path_joblib.exists() and joblib is not None:
            models[name] = joblib.load(path_joblib)
        elif path_pkl.exists():
            with open(path_pkl, "rb") as f:
                models[name] = pickle.load(f)
        else:
            st.error(f"Model {name} not found. Run train.py first.")
            return None
    # load scaler
    if (MODELS_DIR / "scaler.joblib").exists() and joblib is not None:
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    elif (MODELS_DIR / "scaler.pkl").exists():
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        st.error("Scaler not found. Run train.py first.")
        return None
    return models, scaler


def preprocess(df):
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df.get("Timestamp"), errors="coerce")
    df["Hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
    X = df[["Amount", "Hour"]].copy()
    cat_cols = ["TransactionType", "Location", "Merchant"]
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([X, df_cat], axis=1).fillna(0)
    return df, X


def compute_scores(models, scaler, X):
    Xs = scaler.transform(X)
    iso = models["isolation_forest"]
    lof = models["lof"]
    ocsvm = models["one_class_svm"]
    scores = {
        "isolation_forest": -iso.decision_function(Xs),
        "lof": -lof.decision_function(Xs),
        "one_class_svm": -ocsvm.decision_function(Xs),
    }
    return scores


def normalize_scores(arr):
    low, high = np.percentile(arr, [1, 99])
    arr_clipped = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr_clipped)
    return (arr_clipped - low) / (high - low)


def aggregate(scores):
    normed = {k: normalize_scores(v) for k, v in scores.items()}
    stacked = np.vstack([normed[k] for k in normed]).T
    combined = np.mean(stacked, axis=1)
    return normed, combined


def main():
    st.title("Fraud Detection — Anomaly Flagging")
    st.markdown("Upload a CSV of transactions (same columns as training) and get flagged suspicious transactions.")

    models_and_scaler = load_models()
    if models_and_scaler is None:
        st.stop()
    models, scaler = models_and_scaler

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    use_sample = st.checkbox("Use sample dataset (provided)", value=True)

    if uploaded is None and not use_sample:
        st.info("Upload a file or enable sample dataset.")
        st.stop()

    if use_sample and uploaded is None:
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(uploaded)

    st.write(f"Loaded {len(df)} rows")
    df_proc, X = preprocess(df)

    scores = compute_scores(models, scaler, X)
    normed, combined = aggregate(scores)
    df_proc["combined_score"] = combined
    for k, v in normed.items():
        df_proc[f"score_{k}"] = v

    top_n = st.number_input("Top N suspicious transactions", min_value=1, max_value=len(df_proc), value=100)
    flagged = df_proc.sort_values("combined_score", ascending=False).head(top_n)

    st.subheader("Flagged transactions")
    st.dataframe(flagged)

    csv = flagged.to_csv(index=False)
    st.download_button("Download flagged CSV", csv, file_name=f"flagged_top_{top_n}.csv", mime="text/csv")

    st.subheader("Combined score distribution")
    st.bar_chart(df_proc["combined_score"].value_counts(bins=50).sort_index())


if __name__ == "__main__":
    main()
