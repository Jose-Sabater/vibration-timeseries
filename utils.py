import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# file_path = "acceleration.csv"
# time_header = 'Time'
# ts_category = 'acceleration'

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


def read_file(file_path,time_header,ts_category):
    df = pd.read_csv(file_path, header = 1 , parse_dates=[time_header], index_col= time_header)    
    # df=pd.read_csv(file_path, header = 1 , parse_dates=[time_header], index_col= time_header)
    df.columns = [ts_category]
    df[ts_category] = df[ts_category].str.replace(r'[^0-9,.]+','').astype(float)
    acceleration=df[ts_category]
    print(f"Nr of records: {len(acceleration.index)}")
    print(f"Amount of 0's: {acceleration.isnull().sum()}")
    print(f"Values lower than 12: {len(acceleration.loc[acceleration<12])}")
    return df




def first_histogram(df):
    #Histogram with 3 standard deviations
    plt.figure(figsize=(10,5), dpi=100)
    # plt.hist(df['acceleration'], bins = 20
    sns.histplot(df, bins = 20)
    # plt.axvline(x=df['acceleration'].mean()+3*df['acceleration'].std(), color='red',ls= '--')
    # plt.axvline(x=df['acceleration'].mean()-3*df['acceleration'].std(), color='red',ls= '--')
    plt.title('Histogram of all values')
    plt.show()

def transform_df_to_hour(df):
    df_hour = df.resample('H').median()
    df_hour['weekday'] = [d.strftime('%a') for d in df_hour.index]
    df_hour['hour'] = [d.strftime('%H') for d in df_hour.index]
    df_hour['yearweek'] = [d.strftime('%W') for d in df_hour.index]
    df_hour['month'] = [d.strftime('%b') for d in df_hour.index]
    df_hour['yearday'] = [d.strftime('%j') for d in df_hour.index]
    df_hour['year'] = [d.strftime('%Y') for d in df_hour.index]
    df_hour['hour']=df_hour['hour'].astype(int)
    return df_hour


def check_onoff(data_hour):
    data_hour['onoff']= 'off'
    data_hour.loc[data_hour['acceleration']<= 12,'onoff'] = 'off'
    data_hour.loc[data_hour['acceleration']> 12,'onoff'] = 'on'
    return data_hour

def date_filter(df, start_date, end_date):
    sliced_df = df.loc[(df.index < end_date) & (df.index > start_date)]
    return sliced_df

def filter_12(df):
    df=df[df.iloc[:,0]>12]
    return df

def plot_timeseries(df):
    plt.figure(figsize=(16,5), dpi=100)
    plt.plot(df.index, df.iloc[:,0], color='tab:red')
    # plt.axhline(y=[18], color='green', ls='--', lw=2, label='vline_multiple - full height')
    plt.gca().set(title=df.columns[0], xlabel='date', ylabel='m/s2')
    plt.show()

# data = read_file(file_path, time_header, ts_category)
# data_hour = transform_df_to_hour(data)
# data_clean = filter_12(data)
# # first_histogram(data['acceleration'])
# plot_timeseries(data_clean)
# # data = transform_df_to_hour(data)
# # print(data)
def main_function(file_path,time_header,ts_category):
    data = read_file(file_path,time_header,ts_category)
    data_hour = transform_df_to_hour(data)
    data_filtered = filter_12(data_hour)
    data_raw = check_onoff(data_hour)
    print("-------------succesfully loaded data-------------")
    return data_filtered, data_raw


def df_serial_index(df):
    df_new_index= df.copy()
    df_new_index.reset_index(inplace=True)
    rolmean_df = df_new_index.iloc[:,1].rolling(50).mean()
    rolstd_df = df_new_index.iloc[:,1].rolling(50).std()
    higher_bound = rolmean_df + 3 *rolstd_df
    lower_bound = rolmean_df - 3 *rolstd_df
    return df_new_index, rolmean_df, rolstd_df

#On/Off with day filter

def on_off_ratio(df, start_date, end_date):
    df.loc[df['']]

    return onoff_df, 
