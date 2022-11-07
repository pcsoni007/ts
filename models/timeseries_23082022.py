# %%
# Importing required libraries
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from IPython.display import Image
from pmdarima import auto_arima
from dateutil.parser import parse
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# To do remove the graph related libraries .
#import matplotlib.pyplot as plt
# %matplotlib inline
#import seaborn as sns
#import plotly.express as px
#import chart_studio.plotly as ply
#import cufflinks as cf
#import cufflinks as cf
# cf.go_offline()
#cf.set_config_file(offline=False, world_readable=True)


# %% [markdown]
# Need output in form of dataframe, dictinary or json for graph related functions

# %%
# Importing CSV Dataset
file = r"C:\Users\plahare\Downloads\BeerWineLiquor.csv"
# create function that takes a file for reading and creating dataframe


def file_read(data):
    return pd.read_csv(data)


# %%


# %% [markdown]
# # Setting Target col and Date col

# %% [markdown]
# create functions for target column and index column

# %%
# Dropdown for selecting target column for forecasting and date column


def target_col(data, col_name):
    target_column = data[col_name]
    print(target_column.head())

# def ts_col() = 'date'
# Drop down


# %%


# %% [markdown]
# # Set Index

# %%
# Changing date to datetime and set it as an index
def set_index(data, col_name):
    target_col = target_col(col_name)
    data[col_name] = pd.to_datetime(data[col_name])
    data.set_index(col_name, inplace=True)
    # print(data.head())
    return data


# %%
# should have a proper table structure to avoid empty records


# %% [markdown]
# # Shape of Data / Rows & Cols

# %%
# displaying rows and columns #Return dictionary .eg:{"rows": 0, "cols": 0}


def shape_df(data):
    s = data.shape
    s1 = s[0]
    s2 = s[1]
    df = {'Name': ['row_count', 'col_count'], 'Count': [s1, s2]}
    return pd.DataFrame(df)
    # use return here to print dict
    #print('No of rows :{}'.format(s[0]))
    #print('No of Columns:{}'.format(s[1]))


# %%

# %% [markdown]
# # Head & Tail

# %%
# giving choice to user to display head or tail


def display_head_tail(data, choice='Head'):
    if choice == 'Head':
        return data.head()
    elif choice == 'Tail':
        return data.tail()
    else:
        # convert this into dataframe and display
        return {"message": "Invalid choice"}


# %%

# %% [markdown]
# # Describe function

# %%
# display descriptive statistics #Columns rename as 'column_name'


def describe_data(df):
    return df.describe().transpose()


# %%

# %% [markdown]
# # Resampling  Countinous/Discontinous (page 2)

# %%
#df= df.asfreq(pd.infer_freq(df.index))

# %%
# Resampling Function

# %% [markdown]
# # Team is working on this will update you
# def check_Continuity(data):
#     c=pd.infer_freq(data.index)
#     if c==None:
#         print("This is non-continuous data")
#         #Function for Resampling
#     else:
#         print("This is continuous data ")
#         print(c)

# %%
# check_Continuity(df)
import json

def convert_json_to_df(data):
    df = pd.DataFrame(data)
    print(df.transpose())


# %%


# %% [markdown]
# # Null Value Treatment

# %% [markdown]
# ## List of columns having null values


# %%
# This functions creates a dictionary where columns are keys and values are percentage of null values present in that column
def null_list(df):

    mydict = {}  # an empty dictionary for storing null value percentage
    list1 = []
    
    for i in df.columns:
        # this is to create a dictionary with columns which has null values.
        if df[i].isnull().sum() > 0:
            mydict[i] = [(df.isnull().sum())*100 / len(df)][0][i]

    for j, k in mydict.items():
        list1.append(j)

    if len(list1) == 0:
        return {"message": "List does not contain anything."}
    else:
        return mydict

# %% [markdown]
# ## prefer dataframe first then Dictionary then at last json

# %% [markdown]
# ## Graph to display percentage of null values

# %%
# for plotting the null values. this function plots graph of columns in the x-axis and its percentage of null values in the y-axis


def graph(data):

    null_percentage = pd.DataFrame(data.isnull().sum()*100)/len(data)
    # x=[data.columns]#convert this into list
    # y=[null_percentage]# convert this into list
    x = (np.array(data.columns)).tolist()
    # print(x)
    y = (np.array(null_percentage)).tolist()
    # print(y)
    

    my_dict = {
        "x_label": 'Columns',
        "y_label": 'Pecentages',
        "title": "Percentage of null values present in each column",
        "x_value": x,
        "y_value": y,
        "chart_type": 'bar'
    }

    return my_dict


# %%


# %% [markdown]
# ## Null Value Treatment

# %%
# Takes dataframe as an input and returns a dictionary with column names and null %


def get_null_percentages(df):
    mydict = {}

    for key in df.columns:
        mydict[key] = [(df.isnull().sum())*100 / len(df)][0][key]

    return mydict

# %%
# Takes dataframe and column name as an input.


def drop_rows(df, col_name):
    return df.dropna(subset=[col_name], axis=0, how="any", inplace=True)

# %%
# Takes dataframe and column name as an input.


def drop_cols(df, col_name):
    return df.drop([col_name], axis=1, inplace=True)


# %%
# Takes dataframe, column name, and impute method as an input.
def impute(df, col_name, impute_method='interpolation'):

    if df.dtypes[col_name] == str or df.dtypes[col_name] == object:
        return df[col_name].fillna(df[col_name].mode()[0], inplace=True)

    else:
        flag1 = (df[col_name].isnull() & df[col_name].shift(-1).isnull()).any()
        flag2 = df[col_name].head(1).isnull().bool()
        flag3 = df[col_name].tail(1).isnull().bool()

        if flag1 or flag2 or flag3:
            return df[col_name].fillna(df[col_name].interpolate(method='linear', limit_direction="both"), inplace=True)

        elif impute_method == "locf" and (flag1 == False and flag2 == False and flag3 == False):
            return df[col_name].fillna(df[col_name].ffill(), inplace=True)

        elif impute_method == "nocb" and (flag1 == False and flag2 == False and flag3 == False):
            return df[col_name].fillna(df[col_name].bfill(), inplace=True)
