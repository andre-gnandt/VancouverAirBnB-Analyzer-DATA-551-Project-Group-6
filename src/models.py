import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "../data/raw/listings.csv" 

def _clean_money_to_float(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[\$,]", "", regex=True)
         .replace("nan", np.nan)
         .astype(float)
    )

def _clean_response_rate(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[\%,]", "", regex=True)
         .replace("nan", np.nan)
         .astype(float)
    )

TARGET_PRICE = "price"
TARGET_RATE  = "review_scores_rating"  

categorical_cols_price = [
    "property_type",
    "neighbourhood_cleansed"
]

numeric_cols_price = [
    "accommodates", "bedrooms", "beds", "bathrooms", 
     "review_scores_location", "review_scores_rating"
    , "host_response_rate", "review_scores_communication", 
]

FEATURE_COLS_PRICE = numeric_cols_price + categorical_cols_price

categorical_cols_rating = [
    "property_type",
    "neighbourhood_cleansed"
]

numeric_cols_rating = [
    "accommodates", "bedrooms", "beds", "bathrooms", 
    "review_scores_cleanliness", "review_scores_location", 
    "price", "review_scores_communication", "host_response_rate"
]

FEATURE_COLS_RATING = numeric_cols_rating + categorical_cols_rating

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_price = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols_price),
        ("cat", categorical_transformer, categorical_cols_price)
    ],
    remainder="drop"
)

preprocess_rating = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols_rating),
        ("cat", categorical_transformer, categorical_cols_rating)
    ],
    remainder="drop"
)

def _make_model():
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

price_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_price),
    ("model", _make_model())
])

rate_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_rating),
    ("model", _make_model())
])

def train_models(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    df[TARGET_PRICE] = _clean_money_to_float(df[TARGET_PRICE])
    df["host_response_rate"] = _clean_response_rate(df["host_response_rate"])

    df_price = df.dropna(subset=[TARGET_PRICE]).copy()
    Xp = df_price[FEATURE_COLS_PRICE]
    yp = df_price[TARGET_PRICE]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        Xp, yp, test_size=0.2, random_state=42
    )
    price_pipeline.fit(Xp_train, yp_train)
    pred_p = price_pipeline.predict(Xp_test)
    print("PRICE MODEL | MSE:", round(mean_squared_error(yp_test, pred_p), 2),
          "| R2:", round(r2_score(yp_test, pred_p), 3))

    df_rate = df.dropna(subset=[TARGET_RATE]).copy()
    Xr = df_rate[FEATURE_COLS_RATING]
    yr = df_rate[TARGET_RATE]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    rate_pipeline.fit(Xr_train, yr_train)
    pred_r = rate_pipeline.predict(Xr_test)
    print("RATE MODEL  | MSE:", round(mean_squared_error(yr_test, pred_r), 2),
          "| R2:", round(r2_score(yr_test, pred_r), 3))

    return price_pipeline, rate_pipeline

PRICE_MODEL, RATE_MODEL = train_models(CSV_PATH)

def predict_price(input_dict: dict) -> float:
    X = pd.DataFrame([input_dict])[FEATURE_COLS_PRICE]
    return float(PRICE_MODEL.predict(X)[0])

def predict_rate(input_dict: dict) -> float:
    X = pd.DataFrame([input_dict])[FEATURE_COLS_RATING]
    return float(RATE_MODEL.predict(X)[0])