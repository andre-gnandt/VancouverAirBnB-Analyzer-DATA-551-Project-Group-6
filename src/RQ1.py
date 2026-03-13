from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import altair as alt
import os
import math

from dash import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


# =====================================================
#                LOAD DATA
# =====================================================

def find_listings_path():
    paths = [
        "data/clean/listings.csv",
        "data/raw/listings.csv",
        "listings.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("listings.csv not found")

def load_listings():
    df = pd.read_csv("listings.csv")

    # clean price
    df["price"] = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # fill missing ratings
    if df["review_scores_rating"].isna().any():
        df["review_scores_rating"] = df["review_scores_rating"].fillna(
            df["review_scores_rating"].median()
        )

    return df

LISTINGS = load_listings()


# =====================================================
#                HAVERSINE DISTANCE
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2

    return 2 * R * math.asin(math.sqrt(a))


# =====================================================
#              TOURIST HOTSPOTS
# =====================================================

HOTSPOTS = {
    "Downtown": (49.2827, -123.1207),
    "Stanley Park": (49.3043, -123.1443),
    "Gastown": (49.2833, -123.1060),
    "Granville Island": (49.2710, -123.1340)
}

# =====================================================
#                   DASH APP
# =====================================================

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

tourist_tab = dbc.Tab([
    html.H3("RQ1 – Best Airbnb Options for Tourists"),

    html.Div([

        # LEFT PANEL (Filters)
        html.Div([

            html.Label("Tourist Hotspot"),
            dcc.Dropdown(
                id="hotspot",
                options=[{"label":k, "value":k} for k in HOTSPOTS.keys()],
                value="Downtown"
            ),

            html.Br(),

            html.Label("Group Size"),
            dcc.Input(id="group_size", type="number", value=2, min=1),

            html.Br(), html.Br(),

            html.Label("Price Range"),
            dcc.RangeSlider(
                id="price_range",
                min=int(LISTINGS["price"].min()),
                max=int(LISTINGS["price"].quantile(0.99)),
                step=10,
                value=[50, 300]
            ),

            html.Br(),

            html.Label("Max Distance (km)"),
            dcc.Slider(
                id="max_distance",
                min=1,
                max=20,
                step=1,
                value=5
            ),

            html.Br(),

            html.Label("Minimum Rating"),
            dcc.Slider(
                id="min_rating",
                min=0,
                max=5,
                step=0.5,
                value=3
            ),

            html.Br(),

            html.Label("Room Type"),
            dcc.Dropdown(
                id="room_type",
                options=[
                    {"label":"Entire home/apt","value":"Entire home/apt"},
                    {"label":"Private room","value":"Private room"},
                    {"label":"Shared room","value":"Shared room"}
                ],
                multi=True
            ),

            html.Br(),

            html.Label("Essential Amenities"),
            dcc.Dropdown(
                id="amenities",
                options=[
                    {"label":"Wifi","value":"wifi"},
                    {"label":"Kitchen","value":"kitchen"},
                    {"label":"Washer","value":"washer"},
                    {"label":"Dryer","value":"dryer"},
                    {"label":"Elevator","value":"elevator"},
                    {"label":"Wheelchair accessible","value":"wheelchair"}
                ],
                multi=True
            ),

            html.Br(),

            html.Button("Find Best Matches", id="search_btn")

        ], style={"width":"30%","padding":"10px"}),

        # RIGHT PANEL (Results)
        html.Div([

            dash_table.DataTable(
                id="results_table",
                columns=[
                    {"name":"Rank","id":"rank"},
                    {"name":"Name","id":"name"},
                    {"name":"Price","id":"price"},
                    {"name":"Rating","id":"review_scores_rating"},
                    {"name":"Distance (km)","id":"distance"},
                    {"name":"Score","id":"score"}
                ],
                data=[]
            ),

            html.Br(),

            html.Iframe(
                id="map",
                style={"width":"100%","height":"400px","border":"0"}
            )

        ], style={"width":"70%","padding":"10px"})

    ], style={"display":"flex"})
], label="Tourist Listings")


app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab("Price Estimator", label="Price Estimator"),
        dbc.Tab("Rating Estimator", label="Rating Estimator"),
        tourist_tab
    ])
])

@app.callback(
    [Output("results_table","data"),
     Output("map","srcDoc")],
    Input("search_btn","n_clicks"),
    State("hotspot","value"),
    State("group_size","value"),
    State("price_range","value"),
    State("max_distance","value"),
    State("min_rating","value"),
    State("room_type","value"),
    State("amenities","value")
)
def update_results(n, hotspot, group_size, price_range,
                   max_distance, min_rating,
                   room_type, amenities):

    df = LISTINGS.copy()

    # --- distance calculation ---
    center_lat, center_lon = HOTSPOTS[hotspot]
    df["distance"] = df.apply(
        lambda row: haversine(
            row["latitude"], row["longitude"],
            center_lat, center_lon
        ),
        axis=1
    )

    # --- hard filters ---
    df = df[df["accommodates"] >= group_size]
    df = df[(df["price"] >= price_range[0]) &
            (df["price"] <= price_range[1])]
    df = df[df["distance"] <= max_distance]
    df = df[df["review_scores_rating"] >= min_rating]

    if room_type:
        df = df[df["room_type"].isin(room_type)]

    # amenities filter
    if amenities:
        for a in amenities:
            df = df[df["amenities"].str.lower().str.contains(a)]

    if df.empty:
        return [], ""

    # --- scoring (fixed tourist logic) ---
    df["distance_score"] = 1 / (1 + df["distance"])
    df["price_score"] = (df["price"].max() - df["price"]) / \
                        (df["price"].max() - df["price"].min())
    df["rating_score"] = df["review_scores_rating"] / 5
    df["capacity_score"] = np.minimum(1, df["accommodates"]/group_size)

    df["score"] = (
        0.35 * df["distance_score"] +
        0.25 * df["rating_score"] +
        0.20 * df["price_score"] +
        0.10 * df["capacity_score"]
    )

    df = df.sort_values("score", ascending=False).head(10)
    df["rank"] = range(1, len(df)+1)

    table_data = df[[
        "rank","name","price",
        "review_scores_rating",
        "distance","score"
    ]].to_dict("records")

    chart = alt.Chart(df).mark_circle(size=100).encode(
        x="longitude:Q",
        y="latitude:Q",
        color="score:Q",
        tooltip=["name","price","distance"]
    ).properties(height=400)

    return table_data, chart.to_html()


if __name__ == "__main__":
    app.run(debug=True)