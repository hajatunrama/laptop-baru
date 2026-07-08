"""
Training script: Laptop Price Prediction
Membandingkan Linear Regression, Random Forest, dan XGBoost.
Model terbaik disimpan ke models/best_model.pkl untuk dipakai di app Streamlit.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = Path(__file__).parent / "data" / "dataset_laptop_realistis.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Price (USD)"

CATEGORICAL_FEATURES = [
    "Country", "Laptop Brand", "Laptop Model", "CPU Brand",
    "GPU Brand", "GPU Model", "Usage Type",
]
NUMERIC_FEATURES = [
    "CPU Cores", "CPU Threads", "Base Clock (GHz)", "Boost Clock (GHz)",
    "TDP (W)", "RAM (GB)", "Storage (GB)",
    "CPU Performance", "GPU Performance", "Total Performance",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "R2": round(r2_score(y_test, preds), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name}: {metrics}")

    # Pilih model terbaik berdasarkan R2 tertinggi
    best_name = max(results, key=lambda n: results[n]["R2"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nModel terbaik: {best_name} -> {results[best_name]}")

    joblib.dump(best_pipe, MODEL_DIR / "best_model.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    # Simpan juga daftar nilai unik kategori untuk dropdown di Streamlit
    meta = {col: sorted(X[col].unique().tolist()) for col in CATEGORICAL_FEATURES}
    meta["numeric_ranges"] = {
        col: [float(X[col].min()), float(X[col].max())] for col in NUMERIC_FEATURES
    }
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel & metadata tersimpan di folder models/")


if __name__ == "__main__":
    main()
