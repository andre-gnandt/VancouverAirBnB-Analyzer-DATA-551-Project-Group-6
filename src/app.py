import dash
import altair as alt
import pandas as pd
from dash import dash_table
import copy
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "../data/raw/listings.csv" 
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
XpR = Xp_test.rename(columns = {'neighbourhood_cleansed': 'neighborhood', 'property_type':'property type',  'review_scores_rating': 'rating',
'review_scores_location': 'location rating', 'review_scores_communication':'communication', 'host_response_rate': 'response rate'})
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
sorted_feature_names = Xr_test.rename(columns = {'review_scores_cleanliness': 'cleanliness', 'neighbourhood_cleansed': 'neighborhood', 'property_type':'property type', 
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

price_predictors = {"accommodates": 2, "bedrooms": 1, "beds":1, "bathrooms": 1, "host_response_rate":100, "review_scores_communication" : 3,
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

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
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
                    html.Div([
                        html.Label('Host Communication:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='p-communication-input',
                            options=['Excellent', 'Good', 'Average', 'Below Average', 'Poor'],
                            value='Average',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Label('Host Response Rate:', style={'width': '150px', 'margin-right': '10px'}),
                        dcc.Slider(
                            id='p-response-rate-input',
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
                    ], style={'width': '400px', 'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
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
                        html.Label('Accommodates:', style={'width': '120px', 'margin-right': '10px'}),
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
                            value = 1,
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
                            value = 1,
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
                            value = 1,
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
                        html.Iframe(
                            srcDoc=priceChart.to_html(),
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
                        html.Label('Accommodates:', style={'width': '120px', 'margin-right': '10px'}),
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
                            value = 1,
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
                            value = 1,
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
                            value = 1,
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
                        html.Iframe(
                            srcDoc=ratingChart.to_html(),
                            style={'border-width': '0', 'width': '100%', 'height': '330px'}),
                        html.Div([
                        html.H1( id = 'rating', 
                        style = {'text-align': 'center', 'font-size': '85px', 'color': 'red', 'margin-left': '50px', 'margin-top': '5px','height': '100px', 'width': '300px'
                            })
                    ]),
                ], style={'width': '60%', 'padding': '20px'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], label = 'Rating Estimator', style = {'height': '100%'}),
        dbc.Tab('tourist matches', label = 'Tourist Listings')    
    ])
], style = {'height': '100vh'})

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
        Input("p-response-rate-input", 'value'),
        Input("p-communication-input", 'value')
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
    responseRate,
    communication
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
    
    price_predictors['host_response_rate'] = responseRate
    
    if communication == "Poor" : 
        price_predictors['review_scores_communication'] = 1
    elif communication == "Below Average" : 
        price_predictors['review_scores_communication'] = 2
    elif communication == "Average" : 
        price_predictors['review_scores_communication'] = 3
    elif communication == "Good" : 
        price_predictors['review_scores_communication'] = 4
    elif communication == "Excellent" : 
        price_predictors['review_scores_communication'] = 5

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

    return str(pred)+"/5", False

if __name__ == '__main__':
    app.run(debug=True)