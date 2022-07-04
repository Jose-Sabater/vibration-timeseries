from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import main_function ,df_serial_index
import warnings
warnings.filterwarnings('ignore')

file_path = "acceleration.csv"
time_header = 'Time'
ts_category = 'acceleration'

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options



app.layout = html.Div(children=[
    html.H1(children='Timeseries'),

    html.Div(children='''
        Here you will see a display of your data
    '''),

    dcc.Graph(
        id='main-graph',
    ),
    html.P(children= "Select type of graph"),

    dcc.Dropdown(
        id = "graphtype",
        options = ['histogram', 'timeseries', 'box'],
        value = 'histogram',
        clearable = False,
    ),
    html.P(children= "Please set how many standard deviations to use as control limit"),

    dcc.Input(id='stdtolerance', type='number', min=1, max=4, step=1, value=3),
    

    dcc.Graph(
        id='tolerance-graph',
    ),    
    
])

df = main_function (file_path, time_header, ts_category)



@app.callback(
    Output('main-graph', "figure"),
    Input("graphtype", "value"))
def display_graph(graphtype):
    if graphtype == 'histogram':
        fig = px.histogram(df, y=df.iloc[:,0], nbins= 20,
                labels ={"y" : f"{df.columns[0]}"},
                        
                title = 'Histogram')
                
    
    if graphtype == 'timeseries':
        fig = px.line(df, x=df.index, y=df.iloc[:,0],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Time-Series')  
        fig.add_hline(
        y=16, line_width=1, line_dash="dash", 
        line_color="green") 
        fig.add_hline(
        y=13, line_width=1, line_dash="dash", 
        line_color="green")   

    if graphtype == 'box':
        fig = px.box(df, x=df.iloc[:,1], y=df.iloc[:,0], color=df.iloc[:,1],
                labels ={"y" : f"{df.columns[0]}"},
                title = 'Box-Plot')     
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
        name="rolmean"
    )
    df_plot = go.Scatter(
        x=filtered_df.index,
        y=filtered_df.iloc[:,1],
        name=filtered_df.columns[1]
    )
    higher_bound_plot = go.Scatter(
        x=filtered_df.index,
        y=higher_bound,
        name="UCL",

    )
    lower_bound_plot = go.Scatter(
        x=filtered_df.index,
        y=lower_bound,
        name="LCL",

    )
    data = [rolmean_plot, df_plot, higher_bound_plot, lower_bound_plot]
    fig2 = go.Figure(data=data)
    return fig2

                 









if __name__ == '__main__':
    app.run_server(debug=True)