from __future__ import annotations

import os

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dash_table, dcc, html

from src.ml_nn import (
    COMMON_CATEGORICAL,
    COMMON_NUMERIC,
    load_dataset,
    load_or_train_models,
    predict_one,
)

try:
    import altair as alt

    ALTAIR_AVAILABLE = True
except ImportError:  # pragma: no cover
    alt = None
    ALTAIR_AVAILABLE = False


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
        hover_data={"price_num": ":.0f", "review_scores_rating": ":.2f", "neighbourhood_cleansed": True},
        zoom=10,
        height=360,
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def _build_importance_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_values = pd.to_numeric(df[target], errors="coerce")

    for feature in COMMON_NUMERIC:
        feature_values = pd.to_numeric(df[feature], errors="coerce")
        valid = pd.DataFrame({"x": feature_values, "y": target_values}).dropna()
        if len(valid) < 20:
            score = 0.0
        else:
            score = float(abs(valid["x"].corr(valid["y"])))
        rows.append({"feature": feature, "importance": score})

    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(7)


def _build_importance_component(frame: pd.DataFrame, title: str, graph_id: str):
    if ALTAIR_AVAILABLE:
        chart = (
            alt.Chart(frame)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X("importance:Q", title="Absolute Correlation"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(title=title, width=350, height=220)
        )
        return html.Iframe(srcDoc=chart.to_html(), style={"width": "100%", "height": "280px", "border": "0"})

    fig = px.bar(
        frame.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"{title} (Plotly fallback)",
        labels={"importance": "Absolute Correlation", "feature": "Feature"},
    )
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return dcc.Graph(id=graph_id, figure=fig, config={"displayModeBar": False})


# --- Load data + train/load NN models once at startup ---
DATAFRAME = load_dataset()
BACKEND_REQUESTED = os.getenv("NN_BACKEND", "auto")
RATING_BUNDLE, PRICE_BUNDLE, MODEL_METRICS = load_or_train_models(backend=BACKEND_REQUESTED)
DEFAULTS = _make_default_input_values(DATAFRAME)

ROOM_TYPES = sorted(DATAFRAME["room_type"].dropna().astype(str).unique().tolist())
NEIGHBOURHOODS = sorted(DATAFRAME["neighbourhood_cleansed"].dropna().astype(str).unique().tolist())

PRICE_MIN = int(np.nanpercentile(DATAFRAME["price_num"], 5))
PRICE_MAX = int(np.nanpercentile(DATAFRAME["price_num"], 95))

rating_importance = _build_importance_frame(DATAFRAME, target="review_scores_rating")
price_importance = _build_importance_frame(DATAFRAME, target="price_num")

app = dash.Dash(__name__)
server = app.server
app.title = "Vancouver Airbnb Analyzer (Rudimentary NN)"

panel_style = {
    "width": "32%",
    "display": "inline-block",
    "verticalAlign": "top",
    "padding": "12px",
    "border": "1px solid #d2d9e5",
    "borderRadius": "12px",
    "backgroundColor": "#f8fbff",
    "minHeight": "980px",
}

info_style = {"backgroundColor": "#edf5ff", "padding": "8px", "borderRadius": "8px"}

app.layout = html.Div(
    [
        html.H1("Vancouver Airbnb Analyzer"),
        html.P("Milestone 2 prototype: 3 interactive panels and neural-network models (sklearn/PyTorch backend)."),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("RQ1: Tourist Listing Finder"),
                        html.Label("Price Range (CAD/night)"),
                        dcc.RangeSlider(
                            id="rq1-price-range",
                            min=PRICE_MIN,
                            max=PRICE_MAX,
                            step=5,
                            value=[PRICE_MIN, PRICE_MAX],
                            tooltip={"placement": "bottom"},
                        ),
                        html.Label("Minimum Guests", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq1-min-guests", min=1, max=16, step=1, value=2),
                        html.Label("Minimum Rating", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq1-min-rating", min=3.0, max=5.0, step=0.1, value=4.5),
                        html.Label("Room Type(s)", style={"marginTop": "10px"}),
                        dcc.Dropdown(
                            id="rq1-room-types",
                            options=_label_value_options(ROOM_TYPES),
                            value=ROOM_TYPES,
                            multi=True,
                        ),
                        html.Label("Top N", style={"marginTop": "10px"}),
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
                html.Div(
                    [
                        html.H3("RQ2: Predict Guest Rating"),
                        html.P(
                            (
                                f"Backend: {MODEL_METRICS['rating_model']['backend']} | "
                                f"MAE: {MODEL_METRICS['rating_model']['mae']:.3f} | "
                                f"R2: {MODEL_METRICS['rating_model']['r2']:.3f}"
                            ),
                            style=info_style,
                        ),
                        _build_importance_component(
                            rating_importance,
                            title="RQ2: Top Numeric Drivers of Rating",
                            graph_id="rq2-importance-fallback",
                        ),
                        html.Label("Price (CAD/night)"),
                        dcc.Slider(id="rq2-price", min=PRICE_MIN, max=PRICE_MAX, step=5, value=int(DEFAULTS["price_num"])),
                        html.Label("Accommodates", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq2-accommodates", min=1, max=16, step=1, value=int(DEFAULTS["accommodates"])),
                        html.Label("Bedrooms", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq2-bedrooms", min=0, max=10, step=1, value=int(DEFAULTS["bedrooms"])),
                        html.Label("Bathrooms", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq2-bathrooms", min=0.5, max=6.0, step=0.5, value=float(DEFAULTS["bathrooms"])),
                        html.Label("Beds", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq2-beds", min=1, max=12, step=1, value=int(DEFAULTS["beds"])),
                        html.Label("Room Type", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="rq2-room-type", options=_label_value_options(ROOM_TYPES), value=str(DEFAULTS["room_type"])),
                        html.Label("Neighbourhood", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="rq2-neighbourhood", options=_label_value_options(NEIGHBOURHOODS), value=str(DEFAULTS["neighbourhood_cleansed"])),
                        html.Button("Predict Rating", id="rq2-predict", style={"marginTop": "12px"}),
                        html.Div(id="rq2-output", style={"marginTop": "12px", "fontSize": "20px", "fontWeight": "bold"}),
                    ],
                    style=panel_style,
                ),
                html.Div(
                    [
                        html.H3("RQ3: Predict Price"),
                        html.P(
                            (
                                f"Backend: {MODEL_METRICS['price_model']['backend']} | "
                                f"MAE: {MODEL_METRICS['price_model']['mae']:.2f} | "
                                f"R2: {MODEL_METRICS['price_model']['r2']:.3f}"
                            ),
                            style=info_style,
                        ),
                        _build_importance_component(
                            price_importance,
                            title="RQ3: Top Numeric Drivers of Price",
                            graph_id="rq3-importance-fallback",
                        ),
                        html.Label("Review Score Rating"),
                        dcc.Slider(id="rq3-rating", min=1.0, max=5.0, step=0.1, value=float(DEFAULTS["review_scores_rating"])),
                        html.Label("Accommodates", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq3-accommodates", min=1, max=16, step=1, value=int(DEFAULTS["accommodates"])),
                        html.Label("Bedrooms", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq3-bedrooms", min=0, max=10, step=1, value=int(DEFAULTS["bedrooms"])),
                        html.Label("Bathrooms", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq3-bathrooms", min=0.5, max=6.0, step=0.5, value=float(DEFAULTS["bathrooms"])),
                        html.Label("Beds", style={"marginTop": "10px"}),
                        dcc.Slider(id="rq3-beds", min=1, max=12, step=1, value=int(DEFAULTS["beds"])),
                        html.Label("Room Type", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="rq3-room-type", options=_label_value_options(ROOM_TYPES), value=str(DEFAULTS["room_type"])),
                        html.Label("Neighbourhood", style={"marginTop": "10px"}),
                        dcc.Dropdown(id="rq3-neighbourhood", options=_label_value_options(NEIGHBOURHOODS), value=str(DEFAULTS["neighbourhood_cleansed"])),
                        html.Button("Predict Price", id="rq3-predict", style={"marginTop": "12px"}),
                        html.Div(id="rq3-output", style={"marginTop": "12px", "fontSize": "20px", "fontWeight": "bold"}),
                    ],
                    style=panel_style,
                ),
            ],
            style={"display": "flex", "gap": "1%", "justifyContent": "space-between"},
        ),
    ],
    style={"padding": "18px", "fontFamily": "Arial, sans-serif", "backgroundColor": "#f1f5fb"},
)


@app.callback(
    [Output("rq1-map", "figure"), Output("rq1-table", "data")],
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
        return fig, []

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
    return fig, table.to_dict("records")


@app.callback(
    Output("rq2-output", "children"),
    [Input("rq2-predict", "n_clicks")],
    [
        State("rq2-price", "value"),
        State("rq2-accommodates", "value"),
        State("rq2-bedrooms", "value"),
        State("rq2-bathrooms", "value"),
        State("rq2-beds", "value"),
        State("rq2-room-type", "value"),
        State("rq2-neighbourhood", "value"),
    ],
)
def update_rq2(n_clicks, price, accommodates, bedrooms, bathrooms, beds, room_type, neighbourhood):
    if not n_clicks:
        return "Click Predict Rating"

    features = dict(DEFAULTS)
    features.update(
        {
            "price_num": price,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "beds": beds,
            "room_type": room_type,
            "neighbourhood_cleansed": neighbourhood,
        }
    )
    pred = predict_one(RATING_BUNDLE, features)
    return f"Predicted Review Rating: {pred:.2f} / 5.00"


@app.callback(
    Output("rq3-output", "children"),
    [Input("rq3-predict", "n_clicks")],
    [
        State("rq3-rating", "value"),
        State("rq3-accommodates", "value"),
        State("rq3-bedrooms", "value"),
        State("rq3-bathrooms", "value"),
        State("rq3-beds", "value"),
        State("rq3-room-type", "value"),
        State("rq3-neighbourhood", "value"),
    ],
)
def update_rq3(n_clicks, rating, accommodates, bedrooms, bathrooms, beds, room_type, neighbourhood):
    if not n_clicks:
        return "Click Predict Price"

    features = dict(DEFAULTS)
    features.update(
        {
            "review_scores_rating": rating,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "beds": beds,
            "room_type": room_type,
            "neighbourhood_cleansed": neighbourhood,
        }
    )
    pred = predict_one(PRICE_BUNDLE, features)
    return f"Predicted Nightly Price: ${pred:,.0f} CAD"


if __name__ == "__main__":
    app.run(debug=True)
