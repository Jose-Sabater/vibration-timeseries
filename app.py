from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import main_function ,df_serial_index, df_test_train
import warnings
from datetime import date
import base64
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import time
warnings.filterwarnings('ignore')

file_path = "acceleration.csv"
time_header = 'Time'
ts_category = 'acceleration'
image_filename = 'assets/scania_symbol.svg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode() 
app = Dash(__name__)


#Colors
scania_font = 'Scania-Sans'
scania_blue0 = '#001533'
scania_blue1 = '#0F3263'
scania_blue2 = '#2058A8'
scania_blue3 = '#4A89F3'
scania_blue4 = '#87AFE8'
scania_red = '#D6001C'
scania_green = '#1DAB8B'
scania_white = '#F9FAFB'
scania_gray = '#B0B7C4'

df, df_raw = main_function (file_path, time_header, ts_category)
df_sliced = df.copy()

df_test, df_train, acceleration = df_test_train(df, 1300)
print(df_test)
app.layout = html.Div(children=[
    html.Div([
        html.Div([
            html.Img(
                src=app.get_asset_url('scania_symbol.svg').format(encoded_image),
                id = "logo",
                ),
            html.Img(
                src=app.get_asset_url('scania_wordmark_blue_rgb.svg').format(encoded_image),
                id = "wordmark",
                ),                            
            html.H1(children='Timeseries', id= 'Title'),
            ], id = 'Header',
        ),
        
        html.P("Summary of data analysis of your selected dataset"),
        ],
    ),
    html.Div([
        html.H2(children='Data Quality'),
        html.Div(children='''
            Here is some short information of the data you have uploaded
        '''),
        html.Div([
                html.H5('Pie chart of uptime'),
                dcc.Graph(id="pie-graph"),
                html.Hr(),

                ]),

        html.Div(children='''
            Please select the dates you want to do your analysis for:
        '''),    
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed = df.index[0],
            max_date_allowed = df.index[-1],
            initial_visible_month = df.index[0],
            start_date = df.index[0],
            end_date = df.index[-1],
        ),
        html.Button('Reset Filter',id='reset-button', n_clicks=0),
        html.Div(id='output-container-date-picker-range'),
        dcc.Store (id ='sliced-df-value'),
        html.P(children= "Select type of graph"),
        dcc.Dropdown(
            id = "graphtype",
            options = ['histogram', 'timeseries', 'box'],
            value = 'histogram',
            clearable = False,
        ),
        dcc.Graph(
            id='main-graph',
        ),
        html.P(children= "Please set how many standard deviations to use as control limit"),
        dcc.Input(id='stdtolerance', type='number', min=1, max=4, step=1, value=3),
        dcc.Graph(
            id='tolerance-graph',
        ),    
    ]),

    html.Div([
        html.H2(children="Forecasting Preparation"),
        html.P(children=f"Your total dataset is {len(df)} rows long"),
        html.P(children= f"Please select the size you would like for training (recommended is {round(0.85*(len(df)))})"),
        dcc.Input(id='train_size_input', type='number', min=len(df)/2, max=len(df)*0.98, step=1, value=round(len(df)*0.8)),
        dcc.Graph(id='train_test_graph'),
        html.P(children="Select the type of forecast you would like to use. Autoregressive integrated moving average or seasonal"),
        dcc.Dropdown(id='forecast-model',options = ['Arima', 'Sarima'], value="Arima", clearable = False),
        html.H4('Lets choose your trend elements'),
        html.H4('p is your Trend autoregression order'),
        html.H4('d is the Trend difference order'),
        html.H4('q is the Trend moving average order.'),
        html.H4('p-Value:',style={'display':'inline-block','margin-right':20} ),
        dcc.Input(id='p-value', type='number', min=0, max=50, step=1, value=1,style={'display':'inline-block','margin-right':20}),
        html.H4('d-Value:',style={'display':'inline-block','margin-right':20} ),
        dcc.Input(id='d-value', type='number', min=0, max=50, step=1, value=0 ,style={'display':'inline-block','margin-right':20}),
        html.H4('q-Value:',style={'display':'inline-block','margin-right':20} ),
        dcc.Input(id='q-value', type='number', min=0, max=50, step=1, value=0, style={'display':'inline-block','margin-right':20}),
        dcc.Graph(id='fitted_model_graph'),
        html.Div(id='rmse'),
        html.H4('Here are your predictions:'),
        dcc.Graph(id='forecast_plot'),
        html.H4('Please introduce the amount of values you want to predict:',style={'display':'inline-block','margin-right':20} ),
        dcc.Input(id='predictions', type='number',value=100, style={'display':'inline-block','margin-right':20} ),
        dcc.Graph(id='prediction_graph'),

    ])
])

@app.callback(Output('pie-graph', 'figure'),
              Input('pie-graph', 'figure'),)
def populate_graph(dummy):
    pie_graph = px.pie(df_raw,  names= 'onoff', title='Uptime of the machinery',color_discrete_sequence=[scania_blue1,'#B0B7C4'])
    pie_graph.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
    return pie_graph


       


@app.callback(
    Output('my-date-picker-range', 'start_date'),
    Output('my-date-picker-range', 'end_date'),
    Input('reset-button', 'n_clicks'),
    State('my-date-picker-range', 'start_date'),
    State('my-date-picker-range', 'end_date')
    )
def clear_date(n_clicks, start_date, end_date):
    ''' clears the date when button is clicked'''

    if (n_clicks is not None) and (n_clicks > 0):
        # =============================== neither of both following lines work 
        # return ''
        start_date = df.index[0]
        end_date = df.index[-1] 
        return start_date, end_date
    else:
        return start_date, end_date


#Function that stores the df value we want to use in our app (it has to be json serialized to store it)
@app.callback(
    Output('sliced-df-value', 'data'), 
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    )
def change_df_sliced(start_date,end_date):
    df_sliced = df.loc[(df.index < end_date) & (df.index > start_date)]
    return df_sliced.to_json(date_format='iso', orient='split')


# Callback to create the top graphs
@app.callback(
    Output('main-graph', "figure"),
    Input("graphtype", "value"),
    Input('sliced-df-value', 'data'))
def display_graph(graphtype, df_sliced):
    df = pd.read_json(df_sliced, orient='split')
    if graphtype == 'histogram':
        fig = px.histogram(df, y=df.iloc[:,0], nbins= 20,
                labels ={"y" : f"{df.columns[0]}"},                        
                title = 'Histogram',
                color_discrete_sequence=[scania_blue1])
        fig.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
                   
    if graphtype == 'timeseries':
        fig = px.line(df, x=df.index, y=df.iloc[:,0],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Time-Series',
                color_discrete_sequence=[scania_blue1])  
        fig.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
        fig.add_hline(
        y=16, line_width=1, line_dash="dash", 
        line_color="#D6001C") 
        fig.add_hline(
        y=13, line_width=1, line_dash="dash", 
        line_color="#D6001C")   
        fig.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')

    if graphtype == 'box':
        fig = px.box(df, x=df.iloc[:,1], y=df.iloc[:,0], color=df.iloc[:,1],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Box-Plot',
                color_discrete_sequence=[scania_blue0, scania_blue1,scania_blue2, scania_blue3, scania_blue4])
        fig.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')     
    return fig


@app.callback(
    Output('tolerance-graph', "figure"),
    Input('stdtolerance', "value"),
    )

def display_tolerances(stdtolerance):
    filtered_df, rolmean, rolstd = df_serial_index(df)
    higher_bound = rolmean + stdtolerance * rolstd
    lower_bound = rolmean - stdtolerance * rolstd
    rolmean_plot = go.Scatter(
        x=filtered_df.index,
        y=rolmean,
        name="rolmean",
        line_color= scania_green

    )
    df_plot = go.Scatter(
        x=filtered_df.index,
        y=filtered_df.iloc[:,1],
        name=filtered_df.columns[1],
        line_color=scania_blue1
    )
    higher_bound_plot = go.Scatter(
        x=filtered_df.index,
        y=higher_bound,
        name="UCL",
        line_color='#D6001C',
        

    )
    lower_bound_plot = go.Scatter(
        x=filtered_df.index,
        y=lower_bound,
        name="LCL",
        line_color='#D6001C'

    )
    data = [rolmean_plot, df_plot, higher_bound_plot, lower_bound_plot]
    fig2 = go.Figure(data=data)
    fig2.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans',)
    return fig2


@app.callback(
    Output('train_test_graph','figure'),
    Output('fitted_model_graph','figure'),
    Output('forecast_plot', 'figure'),
    Output('rmse','children'),
    Output('prediction_graph', 'figure'),
    Input('train_size_input', 'value'),
    Input('p-value', 'value'),
    Input('d-value', 'value'),
    Input('q-value', 'value'),
    Input('forecast-model', 'value'),
    Input('predictions', 'value')
    )

def train_test(df_size, p, d, q, model,prediction_size):
    df_train, df_test , acceleration= df_test_train(df, df_size)
    fig = px.line(acceleration, color='label', color_discrete_sequence=[scania_blue1,scania_red])

    if model == "Arima":
        start_time = time.time()
        print("starting Arima modeling")
        model=ARIMA(df_train, order = (p,d,q))
        results_model = model.fit()
        end_time = time.time()
        model_final_prediction = ARIMA(acceleration.iloc[:,0], order = (p,d,q))
        print(f"Finished loading Arima {end_time-start_time}")
        print(results_model.loglikelihood_burn, results_model.nobs_diffuse)
        original_plot= go.Scatter(
            x = df_train.index,
            y = df_train,
            name = "Original Model",
            line_color = scania_gray
        )
        fitted_plot = go.Scatter(
            x = df_train.index,
            y = results_model.fittedvalues,
            name = "Fitted model",
            line_color = scania_red
        )

    if model == "Sarima":
        start_time = time.time()
        print("starting Sarima modeling")
        model=SARIMAX(df_train, order = (p,d,q))
        results_model = model.fit()
        print(results_model.loglikelihood_burn, results_model.nobs_diffuse)
        end_time = time.time()
        model_final_prediction = ARIMA(acceleration.iloc[:,0], order = (p,d,q))
        print(f"Finished loading SArima {end_time-start_time}")
        original_plot= go.Scatter(
            x = df_train.index,
            y = df_train,
            name = "Original Model",
            line_color = scania_gray
        )
        fitted_plot = go.Scatter(
            x = df_train.index,
            y = results_model.fittedvalues,
            name = "Fitted model",
            line_color = scania_red
        )

    prediction = results_model.get_forecast(len(df_test.index))
    prediction_df = prediction.conf_int(alpha = 0.05) 
    prediction_df["Predictions"] = results_model.predict(start = prediction_df.index[0], end = prediction_df.index[-1])
    prediction_df.index = df_test.index
    prediction_out = prediction_df["Predictions"]
    rmse = np.sqrt(mean_squared_error(df_test.values, prediction_df["Predictions"]))
    print("RMSE: ",rmse)

    data = [original_plot, fitted_plot]
    fig2 = go.Figure(data=data)

    train_plot = go.Scatter(
        x = df_train.index,
        y = df_train,
        name= 'training data',
        line_color = scania_blue1
    )

    test_plot = go.Scatter(
        x = df_test.index,
        y = df_test,
        name= 'test data',
        line_color = scania_red
    )

    forecast_plot = go.Scatter(
        x = df_test.index,
        y = prediction_out,
        name= 'forecast',
        line_color = scania_green
    )
    data2 = [train_plot, test_plot, forecast_plot]
    forecast_overlay = go.Figure(data=data2)
    print(acceleration.iloc[:,1])
    # Predicting the future

    results_final_prediction= model_final_prediction.fit()
    prediction1 = results_final_prediction.get_forecast(prediction_size)
    prediction_df_1= prediction1.conf_int(alpha = 0.05) 
    print (type(prediction_size))
    prediction_df_1["predictions"] = results_final_prediction.predict(start = prediction_df_1.index[0], end = prediction_df_1.index[-1])
    prediction_df_1.index = pd.RangeIndex(start=acceleration.index[-1], stop=acceleration.index[-1]+prediction_size, step=1)
    prediction_out1 = prediction_df_1["predictions"]

    #Plot
    prediction_plot = go.Scatter(
        x = prediction_out1.index,
        y = prediction_out1,
        name = 'prediction',
        line_color = scania_green
    )
    df_plot = go.Scatter(
        x = acceleration.index,
        y = acceleration.iloc[:,0],
        name = 'original dataset',
        line_color = scania_blue1
    )

    data3 = [prediction_plot, df_plot]
    prediction_plot = go.Figure (data = data3)

    fig.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
    fig2.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
    forecast_overlay.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
    prediction_plot.update_layout(plot_bgcolor = scania_white, paper_bgcolor = scania_white, font_family = 'Scania Sans')
    return fig, fig2, forecast_overlay, html.H5(f"Your RMSE value is: {rmse}"), prediction_plot


#Include also SARIMA, and SARIMAX
#plot autocorrelation function somewhere


if __name__ == '__main__':
    app.run_server(debug=True)