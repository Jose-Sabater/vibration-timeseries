import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

file_path = "acceleration.csv"
time_header = 'Time'
ts_category = 'acceleration'

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

def remove_invalid_data(df):
    df=df[df.iloc[:,0]>12]
    return df



data = read_file(file_path, time_header, ts_category)
data = transform_df_to_hour(data)
data_clean = remove_invalid_data(data)
# first_histogram(data['acceleration'])
print(data)
# data = transform_df_to_hour(data)
# print(data)