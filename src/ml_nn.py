from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "listings.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "nn_only"
RATING_MODEL_PATH = ARTIFACT_DIR / "rating_mlp.joblib"
PRICE_MODEL_PATH = ARTIFACT_DIR / "price_mlp.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

COMMON_NUMERIC = [
    "accommodates",
    "bedrooms",
    "bathrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
]

COMMON_CATEGORICAL = [
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_is_superhost",
    "instant_bookable",
]

RATING_FEATURES = COMMON_NUMERIC + COMMON_CATEGORICAL + ["price_num"]
PRICE_FEATURES = COMMON_NUMERIC + COMMON_CATEGORICAL + ["review_scores_rating"]


def _parse_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_tf(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    lowered = str(value).strip().lower()
    if lowered in {"t", "true", "1", "yes", "y"}:
        return "t"
    if lowered in {"f", "false", "0", "no", "n"}:
        return "f"
    return "Unknown"


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(data_path or DATA_PATH, low_memory=False)
    df["price_num"] = _parse_price(df["price"])

    numeric_cols = list(
        dict.fromkeys(
            COMMON_NUMERIC
            + [
                "review_scores_rating",
                "price_num",
            ]
        )
    )
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["host_is_superhost", "instant_bookable"]:
        df[col] = df[col].map(_normalize_tf)

    for col in COMMON_CATEGORICAL:
        df[col] = df[col].fillna("Unknown").astype(str).replace({"": "Unknown"})

    return df


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_ohe()),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural-network-only Airbnb models.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional dataset path. Defaults to data/raw/listings.csv.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "sklearn", "torch"],
        help="Neural network backend: auto (torch if available), sklearn, or torch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path) if args.data_path else None

if __name__ == "__main__":
    main()
