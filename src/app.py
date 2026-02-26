import dash
import altair as alt
import pandas as pd
from dash import dash_table
import copy
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

price_predictors = {"accommodates": 2, "bedrooms": 1, "bathrooms": 1, "cleanliness": 3, 
"neighborhood": "Arbutus Ridge", "property type": "Boat", "location": "N/A", "rating": "N/A" }

rating_predictors = {"accommodates": 2, "bedrooms": 1, "bathrooms": 1, "cleanliness": 3, 
"neighborhood": "Arbutus Ridge", "property type": "Boat", "location": "N/A", "price":100 }

price_predictors_importance_df = pd.DataFrame({"Listing Feature" : ["property type", "accommodates", "bedrooms", "bathrooms", "cleanliness", "neighborhood", "location rating", "overall rating"],
                                   "Influence %" : [34 , 20, 22, 10, 10, 4, 3, 1] })

rating_predictors_importance_df = pd.DataFrame({"Listing Feature" : ["property type", "accommodates", "bedrooms", "bathrooms", "cleanliness", "neighborhood", "location rating", "price"],
                                   "Influence %" : [34 , 20, 22, 10, 10, 4, 3, 1] })

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

#price values not accurate
rating_predictor_means = {"accommodates" : 3.653825857519789, "price": 140, "cleanliness": 4.754649629018961,
"bedrooms" : 1.6273937625387562, "bathrooms" :  1.342241938974248, "location": 4.815243198680957,}

#price values not accurate
rating_predictor_sds = {"accommodates" : 2.1021891879787655, "price": 200, "cleanliness": 0.43320651404548893,
"bedrooms" : 1.0103229909015374, "bathrooms" :  0.6789179023094165, "location": 0.2761760008990532}

price_predictor_means = {"accommodates" : 3.653825857519789, "cleanliness": 4.754649629018961,
"bedrooms" : 1.6273937625387562, "bathrooms" :  1.342241938974248, "location": 4.815243198680957, "rating": 4.7730358615004125}

price_predictor_sds = {"accommodates" : 2.1021891879787655, "cleanliness": 0.43320651404548893,
"bedrooms" : 1.0103229909015374, "bathrooms" :  0.6789179023094165, "location": 0.2761760008990532, "rating": 0.3982847116779611}

def standardize_price_predictor(predictors, key) :
    if predictors[key] != "N/A":
        predictors[key] = (predictors[key]-price_predictor_means[key])/price_predictor_sds[key]

def standardize_rating_predictor(predictors, key) :
    if predictors[key] != "N/A":
        predictors[key] = (predictors[key]-rating_predictor_means[key])/rating_predictor_sds[key]

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
                        html.Label('Cleanliness:', style={'width': '120px', 'margin-right': '10px'}),
                        dcc.Dropdown(
                            id='p-cleanliness-input',
                            options=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'],
                            value='Average',
                            clearable = False,
                            style={'width': '250px'}
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
                    ##
                    ##Add Amenities to ML models and filters in future milestones
                    ##
                    #html.Div([
                        #html.Label('Amenities:', style={'width': '100px', 'margin-right': '10px'}),
                        #dcc.Dropdown(
                            #id='amenities-input',
                            #options=['Option X', 'Option Y'],
                            #value='Option X',
                            #clearable = False,
                            #multi = True,
                            #style={'width': '300px'}
                        #),
                    #], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
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
                            style={'border-width': '0', 'width': '100%', 'height': '80%'}),
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
                    ##
                    ##Add Amenities to ML models and filters in future milestones
                    ##
                    #html.Div([
                        #html.Label('Amenities:', style={'width': '100px', 'margin-right': '10px'}),
                        #dcc.Dropdown(
                            #id='amenities-input',
                            #options=['Option X', 'Option Y'],
                            #value='Option X',
                            #clearable = False,
                            #multi = True,
                            #style={'width': '300px'}
                        #),
                    #], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),
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
                            style={'border-width': '0', 'width': '100%', 'height': '80%'}),
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
    Input("p-cleanliness-input", 'value')
)
def p_set_cleanliness(value):
    if value == "Poor":
        price_predictors['cleanliness'] = 1
    elif value == "Below Average":
        price_predictors['cleanliness'] = 2
    elif value == "Average":
        price_predictors['cleanliness'] = 3
    elif value == "Good":
        price_predictors['cleanliness'] = 4
    elif value == "Excellent":
        price_predictors['cleanliness'] = 5

@app.callback(
    Input("p-neighborhood-input", 'value')
)
def p_set_neighborhood(value) :
    price_predictors['neighborhood'] = value

@app.callback(
    Input("p-property-type-input", 'value')
)
def p_set_property_type(value):
    price_predictors['property type'] = value

@app.callback(
    Input("p-bedrooms-input", 'value')
)
def p_set_bedrooms(value):
    price_predictors['bedrooms'] = value

@app.callback(
    Input("p-bathrooms-input", 'value')
)
def p_set_bathrooms(value) :
    price_predictors['bathrooms'] = value

@app.callback(
    Input("p-accommodates-input", 'value')
)
def p_set_accommodates(value) :
    price_predictors['accommodates'] = value

@app.callback(
    Input("p-rating-input", 'value')
)
def p_set_rating(value) :
    price_predictors['rating'] = value

@app.callback(
    Input("p-location-input", 'value')
)
def p_set_location(value) :
    price_predictors['location'] = value


@app.callback(
    Input("r-cleanliness-input", 'value')
)
def r_set_cleanliness(value):
    if value == "Poor":
        rating_predictors['cleanliness'] = 1
    elif value == "Below Average":
        rating_predictors['cleanliness'] = 2
    elif value == "Average":
        rating_predictors['cleanliness'] = 3
    elif value == "Good":
        rating_predictors['cleanliness'] = 4
    elif value == "Excellent":
        rating_predictors['cleanliness'] = 5

@app.callback(
    Input("r-neighborhood-input", 'value')
)
def r_set_neighborhood(value) :
    rating_predictors['neighborhood'] = value

@app.callback(
    Input("r-property-type-input", 'value')
)
def r_set_property_type(value):
    rating_predictors['property type'] = value

@app.callback(
    Input("r-bedrooms-input", 'value')
)
def r_set_bedrooms(value):
    rating_predictors['bedrooms'] = value

@app.callback(
    Input("r-bathrooms-input", 'value')
)
def r_set_bathrooms(value) :
    rating_predictors['bathrooms'] = value

@app.callback(
    Input("r-accommodates-input", 'value')
)
def r_set_accommodates(value) :
    rating_predictors['accommodates'] = value

@app.callback(
    Input("r-location-input", 'value')
)
def r_set_location(value) :
    rating_predictors['location'] = value

@app.callback(
    [Output('price', 'children'),
    Output('alert-auto', 'is_open')],
    [Input('calculate-price', 'n_clicks')],
    [State('alert-auto', 'is_open') ]
)
def predict_price(nclicks, is_open) :
    if( nclicks is None or nclicks <= 0) :
        return "", False

    for key in price_predictors :
        if price_predictors[key] is None or price_predictors[key] == "":
            return "", True

    predictors = copy.deepcopy(price_predictors)

    # standardize each input predictor value if necessary
    for key in predictors:
        if key in price_predictor_means.keys():
            standardize_price_predictor(predictors, key)

    #predict value from ML model and return
    total = ""

    for key in predictors :
        total = str(predictors[key])+", "+total

    return total, False

@app.callback(
    [Output('rating', 'children'),
    Output('alert-auto-r', 'is_open')],
    [Input('calculate-rating', 'n_clicks')],
    [State('alert-auto-r', 'is_open') ]
)
def predict_rating(nclicks, is_open) :
    if( nclicks is None or nclicks <= 0) :
        return "", False

    for key in rating_predictors :
        if rating_predictors[key] is None or rating_predictors[key] == "":
            return "", True

    predictors = copy.deepcopy(rating_predictors)

    # standardize each input predictor value if necessary
    for key in predictors:
        if key in rating_predictor_means.keys():
            standardize_rating_predictor(predictors, key)

    #predict value from ML model and return
    total = ""

    for key in predictors :
        total = str(predictors[key])+", "+total

    return total, False

if __name__ == '__main__':
    app.run(debug=True)