import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import seaborn as sns
from dateutil.parser import parse
from pmdarima import auto_arima
import io
import base64
img = io.BytesIO()
import plotly.express as px
import chart_studio.plotly as ply
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# returns rows and cols


def Shape_df(data):
    s = data.shape
    rows = s[0]
    cols = s[1]
    return rows, cols

# returns dataframe


def dtype_conversion(data, date_column):
    data[date_column] = pd.to_datetime(data[date_column])
    return data


def set_index(data, date_col):
    data.set_index(data[date_col], inplace=True)

    return data

# returns list


def rnc(df):
    return (list(df.columns))

# returns dataframe


def D(df):
    return df.describe()


# Outputs Interactive Graph/Charts
# returns url
def plotly_line(data, col_name):
    fig = px.line(x=data.index, y=data[col_name])
    fig.write_image(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

# returns dataframe


def display_head(df):
    df = df.head()
    return df

# returns dataframe


def display_tail(df):
    df = df.tail()
    return df


def graph(data):
    mydict = {}  # an empty dictionary for storing the null values and its percentage
    for i in data.columns:
        mydict[i] = [(data.isnull().sum()) * 100 / len(data)][0][i]
    null_percentage = mydict.values()
    x = (np.array(data.columns)).tolist()
    y = list(null_percentage)
    my_dict = {"x_label": 'Columns',
               "y_label": 'Pecentages',
               "title": "Percentage of null values present in each column",
               "x_value": x,
               "y_value": y,
               "chart_type": 'bar'}
    return my_dict


def data_description(dataframe):
    a = dataframe.describe()
    b = a.transpose()
    d = pd.DataFrame({'col_names': pd.Series(dtype='str'),
                      'col_data_type': pd.Series(dtype='str'),
                      'col_null_val': pd.Series(dtype='int')})
    l1 = []
    l2 = []
    l3 = []
    col = dataframe.columns
    for i in col:
        l1.append(i)
        l2.append(dataframe[i].dtypes)
        l3.append(dataframe[i].isnull().sum())
    d['col_names'] = l1
    d['col_data_type'] = l2
    d['col_null_val'] = l3
    d = d.set_index("col_names", drop=True)
    d.index.name = None
    desc = d.join(b)
    desc.columns.names = ['Column_Name']
    return desc


# returns a dictionary
def null_list(df):
    mydict = {}
    for i in df.columns:
        # this is to create a dictionary with columns which has null values.
        if df[i].isnull().sum() > 0:
            mydict[i] = [(df.isnull().sum()) * 100 / len(df)][0][i]
    return mydict

# returns dataframe


def drop_rows(df, j):
    df = df.dropna(subset=[j], axis=0, how='any', inplace=True)
    return df

# returns dataframe


def drop_column(df, j):
    df = df.drop([j], axis=1, inplace=True)
    return df

# returns dataframe


def impute(df, j):
    if df.dtypes[j] == str or df.dtypes[j] == object:
        df = df[j].fillna(df[j].mode()[0], inplace=True)
    else:
        df = df[j].fillna(df[j].interpolate(
            method='linear', limit_direction="both"), inplace=True)
    return df


# Exhibitng the Unique Different Days Frequencies
def exhhibitFreq(dataframe, col):
    # dataframe = dataframe.reset_index()
    dataframe = dataframe.sort_values(by=col)
    dataframe['Diff_Days'] = dataframe[col] - dataframe[col].shift(1)
    dataframe['Diff_Days'] = pd.to_timedelta(dataframe['Diff_Days'])
    ls = np.unique(dataframe['Diff_Days'])
    dataframe.set_index(col)
    ls = pd.to_timedelta(ls)
    ls = ls.days
    freq = []
    for i in range(len(ls)):
        freq.append(ls[i])
    return freq


def resamplingData1(dataframe, period, col):
    dataframe = dataframe.resample(
        period, on=col).mean().reset_index(drop=False)
    for i in dataframe.columns:
        if dataframe.dtypes[i] == int or dataframe.dtypes[i] == float:
            dataframe[i].interpolate(
                method='linear', limit_direction='both', inplace=True)
        else:
            dataframe[i].interpolate(
                method='bfill', limit_direction="backward", inplace=True)
    return dataframe


# this acf_pacf function
def acf_pacf(series, nlags=15, alpha=0.05):
    acf, ci = sm.tsa.acf(series, nlags=15, alpha=0.05)
    pacf, ci1 = sm.tsa.pacf(series, nlags=15, alpha=0.05)
    mydict_acf = {
        'title': 'ACF_plot',
        'x_label': "Lags",
        'y_label': "PACF_values",
        'x_val': [i for i in range(0, nlags + 2)],
        # y values have two components acf values and confidence interval values.
        'y_val_acf': acf,
        'y_val_confidence_interval': ci,
        'type': 'stem_plot-aka lollipop plots'}
    mydict_pacf = {
        "title": "PACF Plot",
        "x_label": 'Lags',
        "y_label": 'PACF_values',
        "x_values": [j for j in range(0, nlags + 2)],
        "y_values": pacf,
        "y_val_confidence_interval": ci1,
        "Chart_type": "stem_plot-aka lollipop plots"}

    dict1 = {"Interpretation": "ACF represnts auto correlation between varibles w.r.t Time into consideration all components of time series.PACF represnts correlation function of the variables with residuals partially."}
    dict2 = {"": "Both ACF & PACF starts at lag 0 , which is the correlation of variables with itself and therefore results in a correlation of 1. Difference between both is inclusion and exclusion of indirect correlations. Blue area depicts 95% confidence interval."}
    dict3 = {"Sharp Drop Point":
             ["Instant drop lag just after lag 0.",



              "ACF sharp drop point implies MA order & PACF sharp drop point implies AR order",



              "Some basic approach for model choosing are as follows",



              "1. ACF plot declines gradually and PACF drops instantly use AR model.",
              "2. ACF drops instantly and PACF declines gradually use MA model.",
              "3. Both declines gradually use ARMA model",
              "4. Both drops instantly we are not able to model the time series."]}

    dict4 = {"Note":



             "ARIMA and SARIMA models are Intergrated ARMA models we will use the same identified orders from both the plots."}
    return mydict_acf, mydict_pacf, dict1, dict2, dict3, dict4


def stationarity_test3(data, series):
    # .dropna() handles differenced data
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF test statistic', 'p-value',
              '# lags used', 'number_of_observations_used']
    out = pd.Series(result[0:4], index=labels)
    for key, val in result[4].items():
        out[f'critical value ({key})'] = val
    if result[1] < 0.05:
        out1 = "stationary"
    else:
        out1 = "non_stationary"

    statistic, p_value, n_lags, critical_values = kpss(series)
    dict_kpss = {"KPSS test statistic": statistic, "p_value": p_value,
                 "n_lags": n_lags, "critical_values": critical_values}
    #kpss_result=pd.DataFrame(dict_kpss.items(), columns=['KPSS test statistic', 'p_value','n_lags','critical_values'])
    # Format Output

    if p_value < 0.05:
        out2 = "stationary"
    else:
        out2 = "non_stationary"
    if out1 == "stationary" and out2 == "stationary":
        dict_1 = {"Message": "Since both ADF and KPSS test results indicates stationarity,the data is stationary. Kindly proceed further "}
        out_dict = out.to_dict()
        out_dict["Message1"] = "Strong evidence against the null hypothesis"
        out_dict["Message2"] = "Data has no unit root and is stationary"
        dict_kpss["Message11"] = "The data is stationary"
        return out_dict, dict_kpss, dict_1

    elif out1 == "stationary" and out2 == "non_stationary":
        data["diff_1"] = series.diff(periods=1)
        data['diff_1'].dropna()
        dict_2 = {"Message": "KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary.Therefore differencing has been used to make the data stationary."}
        out_dict = out.to_dict()
        out_dict["Message1"] = "Strong evidence against the null hypothesis"
        out_dict["Message2"] = "Data has no unit root and is stationary"
        dict_kpss["Message11"] = "The data is non-stationary"
        second_tar_col = data["diff1"]
        result1 = adfuller(second_tar_col.dropna(), autolag='AIC')
        labels1 = ['ADF test statistic', 'p-value',
                   '# lags used', 'number_of_observations_used']
        output_1 = pd.Series(result[0:4], index=labels)
        for key, val in result1[4].items():
            output_1[f'critical value ({key})'] = val
        if result1[1] < 0.05:
            output_2 = "stationary"
        else:
            output_2 = "non_stationary"
        if output_2 == "stationary":
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}
        return out_dict, dict_kpss, dict_2, final_dict

    elif out1 == "non_stationary" and out2 == "stationary":

        data['data_log'] = np.sqrt(series)
        data['data_diff'] = data['data_log'].diff().dropna()
        dict_2 = {"Message": "KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary.Therefore log transformation has been used to make the data strict stationary."}
        out_dict = out.to_dict()
        out_dict["Message1"] = "Weak evidence against the null hypothesis"
        out_dict["Message2"] = "Data has a unit root and is non-stationary"
        dict_kpss["Message11"] = "The data is stationary"
        second_tar_col = data["data_diff"]
        result1 = adfuller(second_tar_col.dropna(), autolag='AIC')
        labels1 = ['ADF test statistic', 'p-value',
                   '# lags used', 'number_of_observations_used']
        output_1 = pd.Series(result[0:4], index=labels)
        for key, val in result1[4].items():
            output_1[f'critical value ({key})'] = val
        if result1[1] < 0.05:
            output_2 = "stationary"
        else:
            output_2 = "non_stationary"
        if output_2 == "stationary":
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}

        return out_dict, dict_kpss, dict_2, final_dict
    elif out1 == "non_stationary" and out2 == "non_stationary":
        data['data_log'] = np.sqrt(series)
        data['data_diff'] = data['data_log'].diff().dropna()
        dict_2 = {"Message": "Both ADF and KPSS test indicates non-stationarity.Therefore log transformation has been chosen to make the data stationarity"}
        out_dict = out.to_dict()
        out_dict["Message1"] = "Weak evidence against the null hypothesis"
        out_dict["Message2"] = "Data has a unit root and is non-stationary"
        dict_kpss["Message11"] = "The data is non-stationary"
        second_tar_col = data["data_diff"]
        result1 = adfuller(second_tar_col.dropna(), autolag='AIC')
        labels1 = ['ADF test statistic', 'p-value',
                   '# lags used', 'number_of_observations_used']
        output_1 = pd.Series(result[0:4], index=labels)
        for key, val in result1[4].items():
            output_1[f'critical value ({key})'] = val
        if result1[1] < 0.05:
            output_2 = "stationary"
        else:
            output_2 = "non_stationary"
        if output_2 == "stationary":
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict = {"final_message": "The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}
        return out_dict, dict_kpss, dict_2, final_dict
