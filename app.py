from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import main_function ,df_serial_index, df_test_train
import warnings
from datetime import date
import base64
warnings.filterwarnings('ignore')

file_path = "acceleration.csv"
time_header = 'Time'
ts_category = 'acceleration'
image_filename = 'assets/scania_symbol.svg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode() 
app = Dash(__name__)

#General Quality of the data


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
        html.H2(children="Forecasting"),
        html.P(children=f"Your total dataaset is {len(df)} rows long"),
        html.P(children= f"Please select the size you would like for training (recommended is {round(0.85*(len(df)))})"),
        dcc.Input(id='train_size_input', type='number', min=len(df)/2, max=len(df)*0.98, step=100, value=round(len(df)*0.8)),
        dcc.Graph(id='train_test_graph')
    ])
])

@app.callback(Output('pie-graph', 'figure'),
              Input('pie-graph', 'figure'),)
def populate_graph(dummy):
    pie_graph = px.pie(df_raw,  names= 'onoff', title='Up and Down time - Pie Chart',color_discrete_sequence=['#0F3263','#B0B7C4'])
    pie_graph.update_layout(plot_bgcolor = 'rgb(219,250,251)', font_family = 'Scania Sans')
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
                color_discrete_sequence=['#0F3263'])
        fig.update_layout(plot_bgcolor = '#F9FAFB', font_family = 'Scania Sans')
                   
    if graphtype == 'timeseries':
        fig = px.line(df, x=df.index, y=df.iloc[:,0],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Time-Series',
                color_discrete_sequence=['#0F3263'])  
        fig.update_layout(plot_bgcolor = '#F9FAFB', font_family = 'Scania Sans')
        fig.add_hline(
        y=16, line_width=1, line_dash="dash", 
        line_color="#D6001C") 
        fig.add_hline(
        y=13, line_width=1, line_dash="dash", 
        line_color="#D6001C")   
        fig.update_layout(plot_bgcolor = '#F9FAFB', font_family = 'Scania Sans')

    if graphtype == 'box':
        fig = px.box(df, x=df.iloc[:,1], y=df.iloc[:,0], color=df.iloc[:,1],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Box-Plot',
                color_discrete_sequence=['#001533','#0F3263','#2058A8','#4A89F3','#87AFE8'])
        fig.update_layout(plot_bgcolor = '#F9FAFB', font_family = 'Scania Sans')     
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
        line_color= '#1DAB8B'

    )
    df_plot = go.Scatter(
        x=filtered_df.index,
        y=filtered_df.iloc[:,1],
        name=filtered_df.columns[1],
        line_color='#0F3263'
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
    fig2
    return fig2.update_layout(plot_bgcolor = '#F9FAFB', font_family = 'Scania Sans')


@app.callback(
    Output('train_test_graph','figure'),
    Input('train_size_input', 'value')
    )

def train_test(df_size):
    df_train, df_test , acceleration= df_test_train(df, df_size)
    fig = px.line(acceleration, color='label')
    
    return fig
                 
if __name__ == '__main__':
    app.run_server(debug=True)