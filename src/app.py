from __future__ import annotations

import os

import plotly.express as px
from dash import html

#For deploy: replace ml_nn with src.ml_nn
from ml_nn import (
    COMMON_CATEGORICAL,
    COMMON_NUMERIC,
    load_dataset,
)

import dash
import altair as alt
import dash_vega_components as dvc
import pandas as pd
from dash import dash_table, no_update
import copy
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import math
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

CSV_PATH = "data/raw/listings.csv" 
avg_rating = 0
avg_location = 0
avg_price  = 0

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
    #, "host_response_rate", "review_scores_communication", 
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

def _make_rf_model():
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )


def _make_price_model():
    if XGBRegressor is None:
        print("[ML] Price model backend: RandomForestRegressor (fallback: xgboost not installed)")
        return _make_rf_model()

    try:
        print("[ML] Price model backend: XGBRegressor")
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception as exc:
        print(f"[ML] Price model backend: RandomForestRegressor (fallback after xgboost init error: {exc})")
        return _make_rf_model()


def _make_rate_model():
    return _make_rf_model()

price_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_price),
    ("model", _make_price_model())
])

rate_pipeline = Pipeline(steps=[
    ("preprocess", preprocess_rating),
    ("model", _make_rate_model())
])

df = pd.read_csv(CSV_PATH)

df[TARGET_PRICE] = _clean_money_to_float(df[TARGET_PRICE])
df["host_response_rate"] = _clean_response_rate(df["host_response_rate"])

avg_location = df["review_scores_location"].mean()
avg_rating = df["review_scores_rating"].mean()
avg_price = df[TARGET_PRICE].mean()

df_price = df.dropna(subset=[TARGET_PRICE]).copy()
Xp = df_price[FEATURE_COLS_PRICE]
yp = df_price[TARGET_PRICE]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    Xp, yp, test_size=0.2, random_state=42
)
price_pipeline.fit(Xp_train, yp_train)

df_rate = df.dropna(subset=[TARGET_RATE]).copy()
Xr = df_rate[FEATURE_COLS_RATING]
yr = df_rate[TARGET_RATE]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)
rate_pipeline.fit(Xr_train, yr_train)

def predict_price(input_dict: dict) -> float:
    X = pd.DataFrame([input_dict])[FEATURE_COLS_PRICE]
    return float(price_pipeline.predict(X)[0])

def predict_rate(input_dict: dict) -> float:
    X = pd.DataFrame([input_dict])[FEATURE_COLS_RATING]
    return float(rate_pipeline.predict(X)[0])

permutated_importances = permutation_importance(price_pipeline, Xp_test, yp_test, n_repeats=10, random_state=42)
sorted_importances_idx = permutated_importances.importances_mean.argsort()
sorted_importances = permutated_importances.importances_mean[sorted_importances_idx]
XpR = Xp_test.rename(columns = {'accommodates': 'max guests' ,'neighbourhood_cleansed': 'neighborhood', 'property_type':'property type',  'review_scores_rating': 'rating',
'review_scores_location': 'location rating' #, 'review_scores_communication':'communication', 'host_response_rate': 'response rate'
})
sorted_feature_names = XpR.columns[sorted_importances_idx]

percentages = []
i = 0
total = 0
while i < len(sorted_importances) :
    if sorted_importances[i] < 0 :
        sorted_importances[i] = 0
    total = sorted_importances[i] + total
    i=i+1

for val in sorted_importances :
    percentages.append(100*val/total)

price_predictors_importance_df = pd.DataFrame({'Listing Feature': sorted_feature_names, 'Influence %': percentages}).sort_values(by='Influence %', ascending=False)

permutated_importances = permutation_importance(rate_pipeline, Xr_test, yr_test, n_repeats=10, random_state=42)
sorted_importances_idx = permutated_importances.importances_mean.argsort()
sorted_importances = permutated_importances.importances_mean[sorted_importances_idx]
sorted_feature_names = Xr_test.rename(columns = {'accommodates': 'max guests' ,'review_scores_cleanliness': 'cleanliness', 'neighbourhood_cleansed': 'neighborhood', 'property_type':'property type', 
'review_scores_location': 'location rating', 'review_scores_communication':'communication', 'host_response_rate': 'response rate'}).columns[sorted_importances_idx]

percentages = []
i = 0
total = 0
while i < len(sorted_importances) :
    if sorted_importances[i] < 0 :
        sorted_importances[i] = 0
    total = sorted_importances[i] + total
    i=i+1

for val in sorted_importances :
    percentages.append(100*val/total)

rating_predictors_importance_df = pd.DataFrame({'Listing Feature': sorted_feature_names, 'Influence %': percentages}).sort_values(by='Influence %', ascending=False)

price_predictors = {"accommodates": 2, "bedrooms": 1, "beds":1, "bathrooms": 1, #"host_response_rate":100, "review_scores_communication" : 3,
"neighbourhood_cleansed": "Arbutus Ridge", "property_type": "Boat", "review_scores_location": avg_location, "review_scores_rating": avg_rating }

rating_predictors = {"review_scores_cleanliness": 3, "price": avg_price, "accommodates": 2, "bedrooms": 1, "beds":1, "bathrooms": 1, "host_response_rate":100, "review_scores_communication" : 3,
"neighbourhood_cleansed": "Arbutus Ridge", "property_type": "Boat", "review_scores_location": avg_location  }

priceChart = alt.Chart(price_predictors_importance_df).mark_bar().encode(
    x = "Influence %:Q",
    y = alt.Y("Listing Feature:O", sort = '-x')
).properties(
    title = "Top Influences on Price",
    width = 400,
    height = 200
).configure_title(
    fontSize = 36
).configure_axis(
    labelFontSize=18,  
    titleFontSize=24
)

ratingChart = alt.Chart(rating_predictors_importance_df).mark_bar().encode(
    x = "Influence %:Q",
    y = alt.Y("Listing Feature:O", sort = '-x')
).properties(
    title = "Top Influences on Rating",
    width = 400,
    height = 200
).configure_title(
    fontSize = 36
).configure_axis(
    labelFontSize=18,  
    titleFontSize=24
)


def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    filled = values.fillna(values.median())
    span = filled.max() - filled.min()
    scaled = pd.Series(1.0, index=series.index) if span == 0 else (filled - filled.min()) / span
    return 1.0 - scaled if invert else scaled


def _make_default_input_values(df: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for col in COMMON_NUMERIC + ["price_num", "review_scores_rating"]:
        values = pd.to_numeric(df[col], errors="coerce")
        defaults[col] = float(values.median()) if values.notna().sum() else 0.0

    for col in COMMON_CATEGORICAL:
        modes = df[col].mode(dropna=True)
        defaults[col] = str(modes.iloc[0]) if not modes.empty else "Unknown"

    return defaults


def _build_rq1_rankings(
    df: pd.DataFrame,
    price_range: list[float],
    min_guests: int,
    min_rating: float,
    room_types: list[str] | None,
    top_n: int,
) -> pd.DataFrame:
    filtered = df[
        (df["price_num"] >= float(price_range[0]))
        & (df["price_num"] <= float(price_range[1]))
        & (df["accommodates"] >= float(min_guests))
        & (df["review_scores_rating"].fillna(0) >= float(min_rating))
    ].copy()

    if room_types:
        filtered = filtered[filtered["room_type"].astype(str).isin(room_types)]

    if filtered.empty:
        return filtered

    filtered["rating_norm"] = _normalize(filtered["review_scores_rating"])
    filtered["price_norm"] = _normalize(filtered["price_num"], invert=True)
    filtered["reviews_norm"] = _normalize(filtered["number_of_reviews"])

    filtered["tourist_score"] = (
        0.50 * filtered["rating_norm"]
        + 0.30 * filtered["price_norm"]
        + 0.20 * filtered["reviews_norm"]
    )

    return filtered.sort_values("tourist_score", ascending=False).head(int(top_n))


def _label_value_options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _build_map_figure(frame: pd.DataFrame):
    if frame.empty:
        fig = px.scatter_mapbox(
            pd.DataFrame({"latitude": [], "longitude": []}),
            lat="latitude",
            lon="longitude",
            zoom=10,
            height=360,
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            annotations=[
                {
                    "text": "No listings match these filters.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                }
            ],
        )
        return fig

    fig = px.scatter_mapbox(
        frame,
        lat="latitude",
        lon="longitude",
        color="tourist_score",
        size="accommodates",
        hover_name="name",
        labels = {"tourist_score": "Score", "price_num" : "Price Per Night", "review_scores_rating":"Rating", 
        "neighbourhood_cleansed":"Neighborhood", "accommodates": "Max Guests"},
        hover_data={"price_num": ":.2f", "review_scores_rating": ":.2f", "neighbourhood_cleansed": True, "latitude": False,
        "longitude": False},
        zoom=10,
        height=360,
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig

DATAFRAME = load_dataset()
DEFAULTS = _make_default_input_values(DATAFRAME)

ROOM_TYPES = sorted(DATAFRAME["room_type"].dropna().astype(str).unique().tolist())
NEIGHBOURHOODS = sorted(DATAFRAME["neighbourhood_cleansed"].dropna().astype(str).unique().tolist())

PRICE_MIN = int(np.nanpercentile(DATAFRAME["price_num"], 5))
PRICE_MAX = int(np.nanpercentile(DATAFRAME["price_num"], 95))

panel_style = {
    "width": "90%",
    "display": "inline-block",
    "verticalAlign": "top",
    "padding": "12px",
    "border": "1px solid #d2d9e5",
    "borderRadius": "12px",
    "backgroundColor": "#f8fbff",
    "minHeight": "980px",
}

info_style = {"backgroundColor": "#edf5ff", "padding": "8px", "borderRadius": "8px"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Vancouver Airbnb Analyzer"
app.layout = dbc.Container([
    html.H1("Vancouver Airbnb Analyzer"),
    dbc.Tabs([
        dbc.Tab([
            html.Div([
                html.Div([
                    dbc.Alert(
                        "MISSING INPUT VALUES! UNABLE TO CALCULATE PRICE!",
                        id="alert-auto",
                        is_open=False,
                        duration=10000,
                        color="danger",
                        style = {"color": "red",  "font-size": "40px"}
                    ),
                    html.Div([
                        html.Label('Neighborhood:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='p-neighborhood-input',
                            options=['Arbutus Ridge', 'Downtown', 'Downtown Eastside',
                    'Dunbar Southlands', 'Fairview', 'Grandview-Woodland',
                    'Hastings-Sunrise', 'Kensington-Cedar Cottage', 'Kerrisdale',
                    'Killarney', 'Kitsilano', 'Marpole', 'Mount Pleasant', 'Oakridge',
                    'Renfrew-Collingwood', 'Riley Park', 'Shaughnessy', 'South Cambie',
                    'Strathcona', 'Sunset', 'Victoria-Fraserview', 'West End',
                    'West Point Grey'],
                            value='Arbutus Ridge',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}), 
                    html.Div([
                        html.Label('Property Type:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='p-property-type-input',
                            options=['Boat', 'Camper/RV', 'Cave', 'Earthen home', 'Entire bungalow',
                    'Entire condo', 'Entire cottage', 'Entire guest suite',
                    'Entire guesthouse', 'Entire home', 'Entire loft', 'Entire place',
                    'Entire rental unit', 'Entire serviced apartment',
                    'Entire townhouse', 'Entire vacation home', 'Entire villa',
                    'Houseboat', 'Private room in bed and breakfast',
                    'Private room in boat', 'Private room in bungalow',
                    'Private room in camper/rv', 'Private room in condo',
                    'Private room in guest suite', 'Private room in guesthouse',
                    'Private room in home', 'Private room in hostel',
                    'Private room in loft', 'Private room in rental unit',
                    'Private room in resort', 'Private room in tiny home',
                    'Private room in townhouse', 'Private room in villa', 'Riad',
                    'Room in aparthotel', 'Room in bed and breakfast',
                    'Room in boutique hotel', 'Room in hotel', 'Shared room in condo',
                    'Shared room in home', 'Shared room in hostel',
                    'Shared room in hotel', 'Shared room in loft',
                    'Shared room in rental unit', 'Tiny home', 'Tower'],
                            value='Boat',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}), 
                    # html.Div([
                    #     html.Label('Host Communication:', style={'width': '120px', 'margin-right': '10px'}),
                    #     dcc.Dropdown(
                    #         id='p-communication-input',
                    #         options=['Excellent', 'Good', 'Average', 'Below Average', 'Poor'],
                    #         value='Average',
                    #         clearable = False,
                    #         style={'width': '250px'}
                    #     ),
                    # ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    #html.Div([
                        #html.Label('Host Response Rate:', style={'width': '150px', 'margin-right': '10px'}),
                        #dcc.Slider(
                            #id='p-response-rate-input',
                            #min = 0,
                            #max = 100,
                            #step = 5,
                            #value=100,
                            #marks = {
                                #0: '0%',
                                # 25: '25%',
                                # 50: '50%',
                                # 75: '75%',
                                # 100: '100%'
                    #         }
                    #     ),
                    # ], style={'width': '400px', 'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Location Rating:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id = 'p-location-input',
                            options = ["N/A", 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                            value = "N/A",
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Overall Rating:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id = 'p-rating-input',
                            options = ["N/A", 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                            value = "N/A",
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Max Guests:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='p-accommodates-input',
                            type='number',
                            min=1,          
                            step=1,          
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Bedrooms:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='p-bedrooms-input',
                            type='number',
                            min=1,          
                            step=1,         
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Beds:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='p-beds-input',
                            type='number',
                            min=1,          
                            step=1,         
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Bathrooms:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='p-bathrooms-input',
                            type='number',
                            min=0.5,          
                            step=0.5,         
                            placeholder="0.5, 1, 1.5, 2, 2.5...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        dbc.Button(
                            "Calculate Price", 
                            id = 'calculate-price', 
                            style = {'font-size': '24px', 'background-color': 'lightgreen','color': 'green', 'margin-left': '50px', 'margin-top': '5px','height': '60px', 'width': '300px'
                            }
                        ),
                    ]),
                ], style = {'width':'40%', 'padding':'20px'}),
                html.Div([
                        dvc.Vega(
                            spec=priceChart.to_dict(),
                            style={'border-width': '0', 'width': '100%', 'height': '330px'}),
                        html.Div([
                        html.H1( id = 'price', 
                        style = {'text-align': 'center', 'font-size': '85px', 'color': 'red', 'margin-left': '50px', 'margin-top': '5px','height': '100px', 'width': '300px'
                            })
                    ]),
                ], style={'width': '60%', 'padding': '20px'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], label = 'Price Estimator'),
        dbc.Tab([
            html.Div([
                html.Div([
                    dbc.Alert(
                        "MISSING INPUT VALUES! UNABLE TO CALCULATE RATING!",
                        id="alert-auto-r",
                        is_open=False,
                        duration=10000,
                        color="danger",
                        style = {"color": "red",  "font-size": "40px"}
                    ),
                    html.Div([
                        html.Label('Cleanliness:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='r-cleanliness-input',
                            options=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'],
                            value='Average',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Neighborhood:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='r-neighborhood-input',
                            options=['Arbutus Ridge', 'Downtown', 'Downtown Eastside',
                    'Dunbar Southlands', 'Fairview', 'Grandview-Woodland',
                    'Hastings-Sunrise', 'Kensington-Cedar Cottage', 'Kerrisdale',
                    'Killarney', 'Kitsilano', 'Marpole', 'Mount Pleasant', 'Oakridge',
                    'Renfrew-Collingwood', 'Riley Park', 'Shaughnessy', 'South Cambie',
                    'Strathcona', 'Sunset', 'Victoria-Fraserview', 'West End',
                    'West Point Grey'],
                            value='Arbutus Ridge',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}), 
                    html.Div([
                        html.Label('Property Type:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='r-property-type-input',
                            options=['Boat', 'Camper/RV', 'Cave', 'Earthen home', 'Entire bungalow',
                    'Entire condo', 'Entire cottage', 'Entire guest suite',
                    'Entire guesthouse', 'Entire home', 'Entire loft', 'Entire place',
                    'Entire rental unit', 'Entire serviced apartment',
                    'Entire townhouse', 'Entire vacation home', 'Entire villa',
                    'Houseboat', 'Private room in bed and breakfast',
                    'Private room in boat', 'Private room in bungalow',
                    'Private room in camper/rv', 'Private room in condo',
                    'Private room in guest suite', 'Private room in guesthouse',
                    'Private room in home', 'Private room in hostel',
                    'Private room in loft', 'Private room in rental unit',
                    'Private room in resort', 'Private room in tiny home',
                    'Private room in townhouse', 'Private room in villa', 'Riad',
                    'Room in aparthotel', 'Room in bed and breakfast',
                    'Room in boutique hotel', 'Room in hotel', 'Shared room in condo',
                    'Shared room in home', 'Shared room in hostel',
                    'Shared room in hotel', 'Shared room in loft',
                    'Shared room in rental unit', 'Tiny home', 'Tower'],
                            value='Boat',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}), 
                    html.Div([
                        html.Label('Host Communication:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='r-communication-input',
                            options=['Excellent', 'Good', 'Average', 'Below Average', 'Poor'],
                            value='Average',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Host Response Rate:', style={'width': '150px', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='r-response-rate-input',
                            min = 0,
                            max = 100,
                            step = 5,
                            value=100,
                            marks = {
                                0: '0%',
                                25: '25%',
                                50: '50%',
                                75: '75%',
                                100: '100%'
                            }
                        ),
                    ], style={'width': '400px','display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Location Rating:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id = 'r-location-input',
                            options = ["N/A", 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                            value = "N/A",
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Price:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='r-price-input',
                            type='number',
                            min=0,          
                            step=0.01,         
                            placeholder="$122.45",
                            value = avg_price,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Max Guests:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='r-accommodates-input',
                            type='number',
                            min=1,          
                            step=1,          
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Bedrooms:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='r-bedrooms-input',
                            type='number',
                            min=1,          
                            step=1,         
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Beds:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='r-beds-input',
                            type='number',
                            min=1,          
                            step=1,         
                            placeholder="1, 2, 3...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Bathrooms:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Input(
                            id='r-bathrooms-input',
                            type='number',
                            min=0.5,          
                            step=0.5,         
                            placeholder="0.5, 1, 1.5, 2, 2.5...",
                            value = 2,
                            required = True,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        dbc.Button(
                            "Calculate Rating", 
                            id = 'calculate-rating', 
                            style = {'font-size': '24px', 'background-color': 'lightgreen','color': 'green', 'margin-left': '50px', 'margin-top': '5px','height': '60px', 'width': '300px'
                            }
                        ),
                    ]),
                    html.Div(style = {'height': '100px'}),
                ], style = {'width':'40%', 'padding':'20px'}),
                html.Div([
                        dvc.Vega(
                            spec=ratingChart.to_dict(),
                            style={'border-width': '0', 'width': '100%', 'height': '330px'}),
                        html.Div([
                        html.H1( id = 'rating', 
                        style = {'text-align': 'center', 'font-size': '85px', 'color': 'red', 'margin-left': '50px', 'margin-top': '5px','height': '100px', 'width': '300px'
                            })
                    ]),
                ], style={'width': '60%', 'padding': '20px'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], label = 'Rating Estimator', style = {'height': '100%'}),
        dbc.Tab([
            html.Div(
            [
                html.Div(
                    [
                        html.H3("Tourist Listing Finder"),
                        html.Label("Price Range (CAD/night)"),
                        dcc.RangeSlider(
                            id="rq1-price-range",
                            min=PRICE_MIN,
                            max=PRICE_MAX,
                            step=5,
                            value=[PRICE_MIN, PRICE_MAX],
                            tooltip={"placement": "bottom"},
                            allow_direct_input=False,
                        ),
                        html.Label("Minimum Guests", style={"marginTop": "10px"}),
                        dcc.Input(
                            id='rq1-min-guests',
                            type='number',
                            min=1,          
                            step=1,     
                            placeholder="2....",
                            value = 2,
                            required = True,
            
                        ),
                        html.Label("Minimum Rating (1-5)", style={"marginTop": "10px"}),
                        dcc.Input(
                            id='rq1-min-rating',
                            type='number',
                            min=1,          
                            step=0.01,     
                            max = 5,
                            placeholder="4.5/5",
                            value = round(avg_rating, 2),
                            required = True,
                            
                        ),
                        html.Label("Room Type(s)", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="rq1-room-types",
                            options=_label_value_options(ROOM_TYPES),
                            value=ROOM_TYPES,
                            multi=True,
                        ),
                        html.Label("Display N amount of matches:", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq1-top-n", min=5, max=30, step=1, value=10),
                        dcc.Graph(id="rq1-map", style={"marginTop": "10px"}),
                        dash_table.DataTable(
                            id="rq1-table",
                            columns=[
                                {"name": "Name", "id": "name"},
                                {"name": "Neighbourhood", "id": "neighbourhood_cleansed"},
                                {"name": "Room Type", "id": "room_type"},
                                {"name": "Price", "id": "price_display"},
                                {"name": "Rating", "id": "rating_display"},
                                {"name": "Score", "id": "score_display"},
                            ],
                            page_size=8,
                            style_cell={"textAlign": "left", "fontSize": "12px", "padding": "6px"},
                            style_table={"overflowX": "auto"},
                        ),
                    ],
                    style=panel_style,
                ),
            ]
            ),
        ], label = 'Tourist Listings')    
    ])
], style = {'height': '100vh'})

@app.callback(
    Output("p-beds-input", "value", allow_duplicate=True),
    Output("p-bathrooms-input", "value"),
    Output("p-bedrooms-input", "value", allow_duplicate=True),
    Input("p-accommodates-input", "value"),
    State("p-beds-input", "value"),
    State("p-bathrooms-input", "value"),
    State("p-bedrooms-input", "value"),
    prevent_initial_call=True
)
def cascade_on_price_accommodates(accommodates, beds_value, bathrooms_value, bedrooms_value) :
    if accommodates is None:
        raise PreventUpdate

    beds_min = math.ceil(accommodates / 2)
    beds_max = accommodates
    beds_new_value = min(max(beds_value or beds_min, beds_min), beds_max)

    if beds_value == beds_new_value : 
        beds_new_value = no_update

    bathrooms_max = accommodates
    bathrooms_new_value = min(bathrooms_max, bathrooms_value)

    if bathrooms_value == bathrooms_new_value : 
        bathrooms_new_value = no_update

    bedrooms_max = accommodates
    bedrooms_new_value = min(bedrooms_max, bedrooms_value)

    if bedrooms_new_value == bedrooms_value :
        bedrooms_new_value = no_update

    return beds_new_value, bathrooms_new_value, bedrooms_new_value

@app.callback( 
    Output("r-beds-input", "value", allow_duplicate=True),
    Output("r-bathrooms-input", "value"),
    Output("r-bedrooms-input", "value", allow_duplicate=True),
    Input("r-accommodates-input", "value"),
    State("r-beds-input", "value"),
    State("r-bathrooms-input", "value"),
    State("r-bedrooms-input", "value"),
    prevent_initial_call=True
)
def cascade_on_rating_accommodates(accommodates, beds_value, bathrooms_value, bedrooms_value) :
    if accommodates is None:
        raise PreventUpdate

    beds_min = math.ceil(accommodates / 2)
    beds_max = accommodates
    beds_new_value = min(max(beds_value or beds_min, beds_min), beds_max)

    if beds_value == beds_new_value : 
        beds_new_value = no_update

    bathrooms_max = accommodates
    bathrooms_new_value = min(bathrooms_max, bathrooms_value)

    if bathrooms_value == bathrooms_new_value : 
        bathrooms_new_value = no_update

    bedrooms_max = accommodates
    bedrooms_new_value = min(bedrooms_max, bedrooms_value)

    if bedrooms_new_value == bedrooms_value :
        bedrooms_new_value = no_update

    return beds_new_value, bathrooms_new_value, bedrooms_new_value

@app.callback(
    Output("p-accommodates-input", "value", allow_duplicate=True),
    Output("p-bedrooms-input", "value", allow_duplicate=True),
    Input("p-beds-input", "value"),
    State("p-accommodates-input", "value"),
    State("p-bedrooms-input", "value"),
    prevent_initial_call=True
)
def cascade_on_price_beds(beds, accommodates_value, bedrooms) :
    if beds is None:
        raise PreventUpdate

    acc_max = beds*2
    acc_min = beds
    accommodates_new_value = min(max(accommodates_value, acc_min), acc_max)

    if accommodates_new_value == accommodates_value :
        accommodates_new_value = no_update

    bedrooms_max = beds 
    bedrooms_new = min(bedrooms, bedrooms_max)

    if bedrooms_new == bedrooms :
        bedrooms_new = no_update

    return accommodates_new_value, bedrooms_new

@app.callback(
    Output("r-accommodates-input", "value", allow_duplicate=True),
    Output("r-bedrooms-input", "value", allow_duplicate=True),
    Input("r-beds-input", "value"),
    State("r-accommodates-input", "value"),
    State("r-bedrooms-input", "value"),
    prevent_initial_call=True
)
def cascade_on_rating_beds(beds, accommodates_value, bedrooms) :
    if beds is None:
        raise PreventUpdate

    acc_max = beds*2
    acc_min = beds
    accommodates_new_value = min(max(accommodates_value, acc_min), acc_max)

    if accommodates_new_value == accommodates_value :
        accommodates_new_value = no_update

    bedrooms_max = beds 
    bedrooms_new = min(bedrooms, bedrooms_max)

    if bedrooms_new == bedrooms :
        bedrooms_new = no_update

    return accommodates_new_value, bedrooms_new

@app.callback(
    Output("p-accommodates-input", "value", allow_duplicate=True),
    Output("p-beds-input", "value", allow_duplicate=True),
    Input("p-bedrooms-input", "value"),
    State("p-accommodates-input", "value"),
    State("p-beds-input", "value"),
    prevent_initial_call=True
)
def cascade_on_price_bedrooms(bedrooms, accommodates_value, beds) :
    if bedrooms is None:
        raise PreventUpdate

    acc_min = bedrooms
    accommodates_new_value = max(accommodates_value, acc_min)

    if accommodates_new_value == accommodates_value : 
        accommodates_new_value = no_update

    beds_min = bedrooms 
    beds_new = max(beds_min, beds)

    if beds_new == beds :
        beds_new = no_update

    return accommodates_new_value, beds_new

@app.callback(
    Output("r-accommodates-input", "value", allow_duplicate=True),
    Output("r-beds-input", "value", allow_duplicate=True),
    Input("r-bedrooms-input", "value"),
    State("r-accommodates-input", "value"),
    State("r-beds-input", "value"),
    prevent_initial_call=True
)
def cascade_on_rating_bedrooms(bedrooms, accommodates_value, beds) :
    if bedrooms is None:
        raise PreventUpdate

    acc_min = bedrooms
    accommodates_new_value = max(accommodates_value, acc_min)

    if accommodates_new_value == accommodates_value : 
        accommodates_new_value = no_update

    beds_min = bedrooms 
    beds_new = max(beds_min, beds)

    if beds_new == beds :
        beds_new = no_update

    return accommodates_new_value, beds_new

@app.callback(
    Output("p-accommodates-input", "value", allow_duplicate=True),
    Input("p-bathrooms-input", "value"),
    State("p-accommodates-input", "value"),
    prevent_initial_call=True
)
def cascade_on_price_bathrooms(bathrooms, accommodates_value) :
    if bathrooms is None:
        raise PreventUpdate

    acc_min = math.ceil(bathrooms)
    accommodates_new_value = max(accommodates_value, acc_min)

    if accommodates_new_value == accommodates_value :
        accommodates_new_value = no_update

    return accommodates_new_value

@app.callback(
    Output("r-accommodates-input", "value", allow_duplicate=True),
    Input("r-bathrooms-input", "value"),
    State("r-accommodates-input", "value"),
    prevent_initial_call=True
)
def cascade_on_rating_bathrooms(bathrooms, accommodates_value) :
    if bathrooms is None:
        raise PreventUpdate

    acc_min = math.ceil(bathrooms)
    accommodates_new_value = max(accommodates_value, acc_min)

    if accommodates_new_value == accommodates_value :
        accommodates_new_value = no_update

    return accommodates_new_value

@app.callback(
    [
        Input("p-neighborhood-input", 'value'),
        Input("p-property-type-input", 'value'),
        Input("p-bedrooms-input", 'value'),
        Input("p-beds-input", 'value'),
        Input("p-bathrooms-input", 'value'),
        Input("p-accommodates-input", 'value'),
        Input("p-rating-input", 'value'),
        Input("p-location-input", 'value'),
        #Input("p-response-rate-input", 'value'),
        #Input("p-communication-input", 'value')
    ]
)
def set_price_predictors(
    neighborhood, 
    property_, 
    bedrooms,
    beds, 
    bathrooms, 
    accommodates,
    rating,
    location,
    #responseRate,
    #communication
                        ) :
    price_predictors['neighbourhood_cleansed'] = neighborhood
    price_predictors['property_type'] = property_
    price_predictors['bedrooms'] = bedrooms
    price_predictors['beds'] = beds
    price_predictors['bathrooms'] = bathrooms
    price_predictors['accommodates'] = accommodates

    if rating == "N/A":
        price_predictors['review_scores_rating'] = avg_rating
    else:
        price_predictors['review_scores_rating'] = rating

    if location == "N/A":
        price_predictors['review_scores_location'] = avg_location
    else:
        price_predictors['review_scores_location'] = location
    
    #price_predictors['host_response_rate'] = responseRate
    
    #if communication == "Poor" : 
        #price_predictors['review_scores_communication'] = 1
    #elif communication == "Below Average" : 
        #price_predictors['review_scores_communication'] = 2
    #elif communication == "Average" : 
        #price_predictors['review_scores_communication'] = 3
    #elif communication == "Good" : 
        #price_predictors['review_scores_communication'] = 4
    #elif communication == "Excellent" : 
        #price_predictors['review_scores_communication'] = 5

@app.callback(
    [
        Input("r-cleanliness-input", 'value'),
        Input("r-price-input", 'value'),
        Input("r-neighborhood-input", 'value'),
        Input("r-property-type-input", 'value'),
        Input("r-bedrooms-input", 'value'),
        Input("r-beds-input", 'value'),
        Input("r-bathrooms-input", 'value'),
        Input("r-accommodates-input", 'value'),
        Input("r-location-input", 'value'),
        Input("r-response-rate-input", 'value'),
        Input("r-communication-input", 'value')
    ]
)
def set_rating_predictors(
    cleanliness,
    price,
    neighborhood, 
    property_, 
    bedrooms,
    beds, 
    bathrooms, 
    accommodates,
    location,
    responseRate,
    communication
                        ) :
    if cleanliness == "Poor" : 
        rating_predictors['review_scores_cleanliness'] = 1
    elif cleanliness == "Below Average" : 
        rating_predictors['review_scores_cleanliness'] = 2
    elif cleanliness == "Average" : 
        rating_predictors['review_scores_cleanliness'] = 3
    elif cleanliness == "Good" : 
        rating_predictors['review_scores_cleanliness'] = 4
    elif cleanliness == "Excellent" : 
        rating_predictors['review_scores_cleanliness'] = 5

    rating_predictors['price'] = price
    rating_predictors['neighbourhood_cleansed'] = neighborhood
    rating_predictors['property_type'] = property_
    rating_predictors['bedrooms'] = bedrooms
    rating_predictors['beds'] = beds
    rating_predictors['bathrooms'] = bathrooms
    rating_predictors['accommodates'] = accommodates
    
    if location == "N/A":
        rating_predictors['review_scores_location'] = avg_location
    else:
        rating_predictors['review_scores_location'] = location
    
    rating_predictors['host_response_rate'] = responseRate
    
    if communication == "Poor" : 
        rating_predictors['review_scores_communication'] = 1
    elif communication == "Below Average" : 
        rating_predictors['review_scores_communication'] = 2
    elif communication == "Average" : 
        rating_predictors['review_scores_communication'] = 3
    elif communication == "Good" : 
        rating_predictors['review_scores_communication'] = 4
    elif communication == "Excellent" : 
        rating_predictors['review_scores_communication'] = 5


@app.callback(
    [Output("rq1-map", "figure"), Output("rq1-table", "data"),
    Output("rq1-price-range", "min"), Output("rq1-price-range", "max")],
    [
        Input("rq1-price-range", "value"),
        Input("rq1-min-guests", "value"),
        Input("rq1-min-rating", "value"),
        Input("rq1-room-types", "value"),
        Input("rq1-top-n", "value"),
    ],
)
def update_rq1(price_range, min_guests, min_rating, room_types, top_n):
    ranked = _build_rq1_rankings(
        df=DATAFRAME,
        price_range=price_range,
        min_guests=min_guests,
        min_rating=min_rating,
        room_types=room_types,
        top_n=top_n,
    )

    fig = _build_map_figure(ranked)

    if ranked.empty:
        return fig, [], 0, 0

    price_max = ranked['price_num'].max()
    price_min = ranked['price_num'].min()
    
    table = ranked[
        [
            "name",
            "neighbourhood_cleansed",
            "room_type",
            "price_num",
            "review_scores_rating",
            "tourist_score",
        ]
    ].copy()
    table["price_display"] = table["price_num"].map(lambda x: f"${x:,.0f}")
    table["rating_display"] = table["review_scores_rating"].map(lambda x: f"{x:.2f}")
    table["score_display"] = table["tourist_score"].map(lambda x: f"{x:.3f}")
    return fig, table.to_dict("records"), price_min, price_max

@app.callback(
    [Output('price', 'children'),
    Output('alert-auto', 'is_open')],
    [Input('calculate-price', 'n_clicks')],
    [State('alert-auto', 'is_open') ]
)
def calculate_price(nclicks, is_open) :
    if( nclicks is None or nclicks <= 0) :
        return "", False

    for key in price_predictors :
        if price_predictors[key] is None or price_predictors[key] == "":
            return "", True

    #predict value from ML model and return
    pred = round(predict_price(price_predictors), 2)
    if pred < 0 :
        pred = 0.00

    return "$"+str(pred)+"/night", False

@app.callback(
    [Output('rating', 'children'),
    Output('alert-auto-r', 'is_open')],
    [Input('calculate-rating', 'n_clicks')],
    [State('alert-auto-r', 'is_open') ]
)
def calculate_rating(nclicks, is_open) :
    if( nclicks is None or nclicks <= 0) :
        return "", False

    for key in rating_predictors :
        if rating_predictors[key] is None or rating_predictors[key] == "":
            return "", True

    #predict value from ML model and return
    pred = round(predict_rate(rating_predictors), 2)
    if pred > 5 :
        pred = 5
    elif pred < 1:
        pred = 1

    return str(pred)+"/5", False

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=False, host="0.0.0.0", port=8080)
