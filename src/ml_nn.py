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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


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


class TorchRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


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


def _build_sklearn_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        batch_size=128,
        max_iter=350,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _fit_torch_model(X_train: np.ndarray, y_train: np.ndarray) -> TorchRegressor:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend requested but torch is not installed.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchRegressor(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for _epoch in range(80):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model.cpu()


def _predict_torch(model: TorchRegressor, X: np.ndarray) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
    return preds


def _train_single(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    use_log1p_target: bool,
    backend: str,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = df[features + [target]].dropna(subset=[target]).copy()
    X = frame[features]
    y = frame[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    numeric_cols = [
        col
        for col in features
        if col in COMMON_NUMERIC or col in {"price_num", "review_scores_rating"}
    ]
    categorical_cols = [col for col in features if col in COMMON_CATEGORICAL]

    y_train_fit = np.log1p(y_train) if use_log1p_target else y_train

    if backend == "torch":
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
        X_train_mat = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
        X_test_mat = np.asarray(preprocessor.transform(X_test), dtype=np.float32)

        torch_model = _fit_torch_model(X_train_mat, y_train_fit.astype(np.float32))
        pred_test = _predict_torch(torch_model, X_test_mat)

        bundle_model = {
            "preprocessor": preprocessor,
            "state_dict": torch_model.state_dict(),
            "input_dim": int(X_train_mat.shape[1]),
        }
        backend_used = "torch"
    else:
        pipeline = _build_sklearn_pipeline(numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train_fit)
        pred_test = pipeline.predict(X_test).reshape(-1)
        bundle_model = pipeline
        backend_used = "sklearn"

    if use_log1p_target:
        pred_test = np.expm1(pred_test)
        pred_test = np.maximum(pred_test, 0.0)
    else:
        pred_test = np.clip(pred_test, 1.0, 5.0)

    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))

    return {
        "model": bundle_model,
        "backend": backend_used,
        "features": features,
        "target": target,
        "use_log1p_target": use_log1p_target,
        "metrics": {
            "mae": mae,
            "r2": r2,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "backend": backend_used,
        },
    }


def train_and_save_models(data_path: Path | None = None, backend: str = "auto") -> dict[str, Any]:
    df = load_dataset(data_path=data_path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    selected_backend = backend
    if backend == "auto":
        selected_backend = "torch" if TORCH_AVAILABLE else "sklearn"
    if selected_backend == "torch" and not TORCH_AVAILABLE:
        selected_backend = "sklearn"

    rating_bundle = _train_single(
        df=df,
        target="review_scores_rating",
        features=RATING_FEATURES,
        use_log1p_target=False,
        backend=selected_backend,
    )
    price_bundle = _train_single(
        df=df,
        target="price_num",
        features=PRICE_FEATURES,
        use_log1p_target=True,
        backend=selected_backend,
    )

    joblib.dump(rating_bundle, RATING_MODEL_PATH)
    joblib.dump(price_bundle, PRICE_MODEL_PATH)

    metrics = {
        "requested_backend": backend,
        "rating_model": rating_bundle["metrics"],
        "price_model": price_bundle["metrics"],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model_bundle(target: str) -> dict[str, Any]:
    if target == "rating":
        return joblib.load(RATING_MODEL_PATH)
    if target == "price":
        return joblib.load(PRICE_MODEL_PATH)
    raise ValueError("target must be 'rating' or 'price'")


def load_or_train_models(
    data_path: Path | None = None,
    backend: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (RATING_MODEL_PATH.exists() and PRICE_MODEL_PATH.exists() and METRICS_PATH.exists()):
        metrics = train_and_save_models(data_path=data_path, backend=backend)
    else:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    return load_model_bundle("rating"), load_model_bundle("price"), metrics


def _predict_from_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if bundle["backend"] == "torch":
        if not TORCH_AVAILABLE:
            raise RuntimeError("This model was trained with torch, but torch is not installed.")
        preprocessor = bundle["model"]["preprocessor"]
        input_dim = int(bundle["model"]["input_dim"])
        state_dict = bundle["model"]["state_dict"]
        model = TorchRegressor(input_dim=input_dim)
        model.load_state_dict(state_dict)
        X = np.asarray(preprocessor.transform(frame), dtype=np.float32)
        return _predict_torch(model, X)

    pipeline = bundle["model"]
    return pipeline.predict(frame).reshape(-1)


def predict_one(bundle: dict[str, Any], input_values: dict[str, Any]) -> float:
    features = bundle["features"]
    frame = pd.DataFrame([{feature: input_values.get(feature) for feature in features}])
    prediction = float(_predict_from_bundle(bundle, frame)[0])

    if bundle["use_log1p_target"]:
        prediction = float(np.expm1(prediction))
        prediction = max(prediction, 0.0)
    else:
        prediction = float(np.clip(prediction, 1.0, 5.0))
    return prediction


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
    metrics = train_and_save_models(data_path=data_path, backend=args.backend)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
