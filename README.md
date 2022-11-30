# Vibration and Temperature Data -timeseries

## Description
A project to analyze time series data from vibration and temperature sensors. The idea is to do an exploration of the data and the visualize it.  
It is further completed with some forecasting  
There is also a dash app built within to visualize all results for the non-programmer.

### Exploration  
![plot](./assets/histogram.png)  
![plot](./assets/boxplot.png) 
![plot](./assets/violin_charts.png) 
### Forecasting
![plot](./assets/test_train.png)
#### Arima
![plot](./assets/fitted_model.png)
#### Sarima
![plot](./assets/fitted_model_sarima.png)
#### Results
![plot](./assets/forecast_sarima.png)
![plot](./assets/forecast_alldata.png)

## Usage
acceleration.ipynb is a thorough exploration of the data including ARIMA and SARIMA forecasting
run app.py -- is the Dash application which feeds from utils and currently acceleration.csv  

## Roadmap

- [x] Exploration completed
- [x] Forecasting done for the next 30 days to the dataset
- [ ] Complete a web browser application to use by none programmers


## Authors and acknowledgment
José María Sabater

## License
Open

## Project status
Work in progress
