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
import math
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

import plotly.graph_objects as go 



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


def drop_rows(df, col_name):
    return df.dropna(subset=[col_name], axis=0, how="any", inplace=True)

def drop_cols(df, col_name):
    return df.drop([col_name],axis=1,inplace=True)

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
            return df[col_name].fillna(df[col_name].ffill(),inplace=True)
        elif impute_method == "nocb" and (flag1 == False and flag2 == False and flag3 == False):
            return df[col_name].fillna(df[col_name].bfill(),inplace=True)


# exhibit frequency of data
def exhhibitFreq(dataframe, col, dtformat):
    dataframe = dataframe.reset_index()
#     dataframe = dataframe.sort_values(by=col)
    dataframe[col] = pd.to_datetime(dataframe[col], format=dtformat) #to convert date column into datetime datatype
    dataframe['Diff_Days'] = dataframe[col] - dataframe[col].shift(1)
    dataframe['Diff_Days'] = pd.to_timedelta(dataframe['Diff_Days'])
    ls = np.unique(dataframe['Diff_Days'])
    dataframe.set_index(col)
    ls = pd.to_timedelta(ls)
    ls = ls.days
    freq = []
    for i in range(len(ls)):
        print(str(ls[i]), str(ls[i]).isdigit())
        if math.isnan(ls[i]):
            freq.append(str(ls[i]))
        else:
            freq.append(ls[i])

    return freq


# # Exhibitng the Unique Different Days Frequencies
# def exhhibitFreq(dataframe, col):
#     # dataframe = dataframe.reset_index()
#     dataframe = dataframe.sort_values(by=col)
#     dataframe['Diff_Days'] = dataframe[col] - dataframe[col].shift(1)
#     dataframe['Diff_Days'] = pd.to_timedelta(dataframe['Diff_Days'])
#     ls = np.unique(dataframe['Diff_Days'])
#     dataframe.set_index(col)
#     ls = pd.to_timedelta(ls)
#     ls = ls.days
#     freq = []
#     for i in range(len(ls)):
#         freq.append(ls[i])
#     return freq


def resamplingData(dataframe, period, col, dtformat):
#     dataframe = dataframe.reset_index()
    dataframe[col] = pd.to_datetime(dataframe[col], format=dtformat)
    dataframe = dataframe.resample(period, on=col).mean().reset_index(drop=False)
    for i in dataframe.columns:
        if dataframe.dtypes[i] == int or dataframe.dtypes[i] == float:
            dataframe[i].interpolate(method='linear', limit_direction='both', inplace=True)
        else:
            dataframe[i].interpolate(method='bfill', limit_direction="backward", inplace=True)

    return dataframe.set_index(col, drop=True)


# ================ EDA Functions ================= #

def plot_vs_graph(df,target_col):
    
    my_dict={"x_label": 'Time',
             "y_label": target_col,
             "title": target_col+" w.r.t. time",
             "x_value":df.reset_index()[df.index.name].to_string(index=False).split('\n'),
             "y_value":df.reset_index()[target_col].to_string(index=False).split('\n'),
             "chart_type":'lineplot',
             "interpretation" : "This graph represents visualization of dependent or target variable w.r.t Time.This depicts how the dependent variable varies with the time. X axis represents time and Y axis represents dependent variable. "
             }
    return my_dict

#This function is from UI perspective
def resample_plot(data,target_col,resample_alias="M"):
    A=pd.DataFrame(data[target_col].resample(resample_alias).max())
    
    my_dict={
        "title":"Resampled Graph Of Target Column",
        "x_label":'Date',
        "y_label": target_col,
        "x_value": A.reset_index()[A.index.name].to_string(index=False).split('\n'), #A.index,
        "y_value": A.reset_index()[target_col].to_string(index=False).split('\n'), #A.values,
        "chart_type":"BarPlot"}
    return my_dict

# displaying top n values in dataframe
def top_n_values(df,target_col,n):
    top_values = pd.DataFrame(df[target_col].sort_values(ascending=False).head(int(n)))
    return top_values

def plot_top_n(data,target_col, top_n):
    df = top_n_values(data,target_col,top_n)
    my_dict={
        "title":"Visualization of Top "+ top_n +" values Of Target Variable",
        "x_label":'Date',
        "y_label": target_col,
        "x_value":df.reset_index()[df.index.name].to_string(index=False).split('\n'),
        "y_value":df.reset_index()[target_col].to_string(index=False).split('\n'),
        "chart_type":"BarPlot",
        "interpretation":"\n This graph represents visualization of Top "+ top_n +" values of dependent or target variable w.r.t Time. X axis represents time and Y axis represents top values of dependent variable. "
        }
    return my_dict



# ================ Stationarity Functions ================= #

# seasonal decomposition plot
def decomposition(target_col,choice='A'):
    y = target_col.to_frame()

    # Multiplicative Decomposition
    if choice == 'M':   
        result=seasonal_decompose(y, model='multiplicative')       
        Date= list(y.reset_index()['date'].to_string(index=False).split('\n'))  # y.index

        observed = result.observed
        Trend_comp= result.trend
        Seasonal_comp= result.seasonal
        Residual_comp= result.resid
        
        observed_val = list(observed.to_string(index=False).split('\n'))[1:]
        trend_val = list(Trend_comp.to_string(index=False).split('\n'))[1:]
        seasonal_val = list(Seasonal_comp.to_string(index=False).split('\n'))[1:]
        residual_val = list(Residual_comp.to_string(index=False).split('\n'))[1:]

        # list of [x,y] values for scatter chart
        rs_list = []
        for index in range(len(residual_val)):
            if "NaN" in residual_val[index]:
                rs_list.append([Date[index], 0])
            else:
                rs_list.append([Date[index], residual_val[index]])


        observed_graph = {
            "title":"Multiplicative Decompose",
            "x_label":'Date',
            "y_label": 'Observed',
            "x_values": Date,
            "y_values": observed_val,
            "chart_type":"line"
            }
    
    
        trend_graph = {
            "x_label":'Date',
            "y_label": 'Trend',
            "x_values": Date,
            "y_values": trend_val,
            "chart_type":"line"
            }
      
    
        seasonal_graph = {
            "x_label":'Date',
            "y_label": 'Seasonal',
            "x_values": Date,
            "y_values": seasonal_val,
            "chart_type":"line"
            }
    
        residual_graph = {
            "x_label":'Date',
            "y_label": 'Residual',
            "x_values": Date,
            "y_values": residual_val, #rs_list,
            "chart_type":"line"
        }
        
        mydict = {
                "interpretation" : "Here X axis represents Time and Y axis represents Normal scaled data. Time series has 4 components Trend,seasonality,cyclical variation and irregular variation.",
                "trend_component": "This is useful in predicting future movements. Over a long period of time, the trend shows whether the data tends to increase or decrease",
                "seasonal_component": "The seasonal component of a time series is the variation in some variable due to some predetermined patterns in its behavior.",
                "cyclical_component": "The cyclical component in a time series is the part of the movement in the variable which can be explained by other cyclical movements in the economy.",
                "irregular_component": "this term gives information about non-seasonal patterns.",
                "note":"Time series has two types of decomposition models Additive Model and Multiplicative model. The plot shows the decomposition of your time series data in its seasonal component, its trend component and the remainder. If you add or multiply the decomposition together you would get back the actual data. First block represents original series , second represents trend , third represents seasonality presents, fourth represents error component or residual.For additive if we add below three blocks we get original data series. Similarly for multiplicative we have to multiply the components.",
                "observed_graph" : observed_graph,
                "trend_graph" : trend_graph,
                'seasonal_graph' : seasonal_graph,
                'residual_graph' : residual_graph,
            }

        return mydict

    elif choice == 'A':
        
    # Additive Decomposition
        result=seasonal_decompose(y, model='additive')
        
        Date= y.reset_index()['date'].to_string(index=False).split('\n')  # y.index
        observed = result.observed
        Trend_comp= result.trend
        Seasonal_comp= result.seasonal
        Residual_comp= result.resid

        observed_val = observed.to_string(index=False).split('\n')[1:]
        trend_val = Trend_comp.to_string(index=False).split('\n')[1:]
        seasonal_val = Seasonal_comp.to_string(index=False).split('\n')[1:]
        residual_val = Residual_comp.to_string(index=False).split('\n')[1:]

        # list of [x,y] values for scatter chart
        rs_list = []
        for index in range(len(residual_val)):
            if "NaN" in residual_val[index]:
                rs_list.append([Date[index], 0])
            else:
                rs_list.append([Date[index], residual_val[index]])


        observed_graph = {
            "title":"Additive Decompose",
            "x_label":'Date',
            "y_label": 'Observed',
            "x_values": Date,
            "y_values":observed_val,
            "chart_type":"line"
            }
    
    
        trend_graph = {
            "x_label":'Date',
            "y_label": 'Trend',
            "x_values": Date,
            "y_values": trend_val,
            "chart_type":"line"
            }
    
        seasonal_graph = {
            "x_label":'Date',
            "y_label": 'Seasonal',
            "x_values": Date,
            "y_values": seasonal_val,
            "chart_type":"line"
        }
    
        residual_graph = {
            "x_label":'Date',
            "y_label": 'Residual',
            "x_values": Date,
            "y_values": residual_val,#rs_list, 
            "chart_type":"scatter"
            }
        

        mydict = {
                "interpretation" : "Here X axis represents Time and Y axis represents Normal scaled data. Time series has 4 components Trend,seasonality,cyclical variation and irregular variation.",
                "trend_component": "This is useful in predicting future movements. Over a long period of time, the trend shows whether the data tends to increase or decrease",
                "seasonal_component": "The seasonal component of a time series is the variation in some variable due to some predetermined patterns in its behavior.",
                "cyclical_component": "The cyclical component in a time series is the part of the movement in the variable which can be explained by other cyclical movements in the economy.",
                "irregular_component": "This term gives information about non-seasonal patterns.",
                "note":"Time series has two types of decomposition models Additive Model and Multiplicative model. The plot shows the decomposition of your time series data in its seasonal component, its trend component and the remainder. If you add or multiply the decomposition together you would get back the actual data. First block represents original series , second represents trend , third represents seasonality presents, fourth represents error component or residual.For additive if we add below three blocks we get original data series. Similarly for multiplicative we have to multiply the components.",
                "observed_graph" : observed_graph,
                "trend_graph" : trend_graph,
                'seasonal_graph' : seasonal_graph,
                'residual_graph' : residual_graph,
            }

        return mydict


# Plot for checking stationarity
def stationarity_check_dict(df,target_col):
    #Determing rolling statistics
    rolmean = df.rolling(12).mean()
    rolstd = df.rolling(12).std()
    
    ind = df.index.name
    df_x= list(df.reset_index()[ind].to_string(index=False).split('\n'))
    df_y= df[target_col].to_string(index= False).split('\n')[1:]

    x_rollmean=rolmean.reset_index()[rolmean.index.name].to_string(index=False).split('\n') 
    y_rollmean=rolmean[target_col].to_string(index= False).split('\n')[1:]
    
    x_rolstd = rolstd.reset_index()[rolstd.index.name].to_string(index=False).split('\n') 
    y_rolstd = rolstd[target_col].to_string(index= False).split('\n')[1:]
    
    
    mydict_df={ 
        "title":"Stationarity check Plot",
        "x_label" :'Date',
        "y_label" :target_col,
        "legends" :['Original'],
        "x_values":df_x,
        "y_values":df_y,
        "Chart_type":"line"
    }
   
    mydict_rollmean={ 
        "x_label":'Date',
        "y_label": 'RollingMean',
        "legends":['RollingMean'],
        "x_values": x_rollmean,
        "y_values":y_rollmean,
        "Chart_type":"line"}

    mydict_rollstd = {
        "x_label":'Date',
        "y_label": 'RollingStd',
        "legends":['RollingStd'],
        "x_values": x_rolstd,
        "y_values": y_rolstd,
        "Chart_type":"line"}
    
    mDict={ 
             "stationarity": " Stationarity means that the statistical properties of a process generating a time series do not change over time. That is Mean and Standard deviation is approximately constant over time.\n\nStationarity Graph represents stationarity of the series w.r.t. Time. X axis depicts time and Y axis depicts Dependent variable . Blue line represents the original Time series data , Red line represents Mean of the series data and Black line represents standard deviation of the series. ",
             "dataframe" : mydict_df,
             "rollmean" : mydict_rollmean,
             "rollstd" : mydict_rollstd
            } 
    
    return mDict
    

def stationarity_test(data,series):
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data  
    labels = ['adf_test_stat','p_value','lags_used','number_of_observations_used']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        l = list(key)
        l.remove('%')
        k = "".join(l)
        print(k)
        out[f'critical_value_{k}']=val
    if result[1] < 0.05:        
        out1="stationary"
    else:
        out1="non_stationary"
        
    statistic, p_value, n_lags, critical_values = kpss(series)

    print('Critical Values', critical_values)
    dict_kpss={"kpss_test_stat":statistic,"p_value":p_value,"lags_used":n_lags,"critical_values":critical_values}
    #kpss_result=pd.DataFrame(dict_kpss.items(), columns=['KPSS test statistic', 'p_value','n_lags','critical_values'])
    # Format Output
    
    if p_value<0.05:
        out2="stationary"
    else:
        out2="non_stationary"
    if out1=="stationary" and out2=="stationary":
        dict_1={"msg":"Since both ADF and KPSS test results indicates stationarity,the data is stationary. Kindly proceed further "}
        out_dict=out.to_dict()
        out_dict["msg1"]="Strong evidence against the null hypothesis"
        out_dict["msg2"]="Data has no unit root and is stationary"
        dict_kpss["msg"]="The data is stationary"
        mdict = { "adf_test": out_dict, "kpss_test": dict_kpss, "final_msg": dict_1 }
        
        return mdict
        
    elif out1=="stationary" and out2=="non_stationary":
        data["diff_1"] =series.diff(periods=1)
        data['diff_1'].dropna()
        dict_2={"msg":"KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary.Therefore differencing has been used to make the data stationary."}
        out_dict=out.to_dict()
        out_dict["msg1"]="Strong evidence against the null hypothesis"
        out_dict["msg2"]="Data has no unit root and is stationary"
        dict_kpss["msg"]="The data is non-stationary"
        second_tar_col=data["diff_1"]
        result1=adfuller(second_tar_col.dropna(),autolag='AIC')
        labels1 = ['adf_test_stat','p_value','lags_used','number_of_observations_used']
        output_1 = pd.Series(result[0:4],index=labels)
        for key,val in result1[4].items():
            l = list(key)
            print(l)
            l.remove('%')
            k = "".join(l)
            print(k)
            output_1[f'critical_value_{k}']=val
        if result1[1] < 0.05:
            output_2="stationary"
        else:
            output_2="non_stationary"
        if output_2=="stationary":
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}
        mdict = { "adf_test": out_dict, "kpss_test": dict_kpss, "conversion_dict": dict_2, "final_test": final_dict }
        return mdict
    elif out1=="non_stationary" and out2=="stationary":
        
        data['data_log']=np.sqrt(series)
        data['data_diff']=data['data_log'].diff().dropna()
        dict_2={"msg":"KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary.Therefore log transformation has been used to make the data strict stationary."}
        out_dict=out.to_dict()
        out_dict["msg1"]="Weak evidence against the null hypothesis"
        out_dict["msg2"]="Data has a unit root and is non-stationary"
        dict_kpss["msg"]="The data is stationary"
        second_tar_col=data["data_diff"]
        result1=adfuller(second_tar_col.dropna(),autolag='AIC')
        labels1 = ['adf_test_stat','p_value','lags_used','number_of_observations_used']
        output_1 = pd.Series(result[0:4],index=labels)
        for key,val in result1[4].items():
            l = list(key)
            print(l)
            l.remove('%')
            k = "".join(l)
            print(k)
            output_1[f'critical_value_{k}']=val
        if result1[1] < 0.05:
            output_2="stationary"
        else:
            output_2="non_stationary"
        if output_2=="stationary":
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}
        mdict = { "adf_test": out_dict, "kpss_test": dict_kpss, "conversion_dict": dict_2, "final_test": final_dict }
        return mdict
    elif out1=="non_stationary" and out2=="non_stationary":
        data['data_log']=np.sqrt(series)
        data['data_diff']=data['data_log'].diff().dropna()
        dict_2={"msg":"Both ADF and KPSS test indicates non-stationarity.Therefore log transformation has been chosen to make the data stationarity"}
        out_dict=out.to_dict()
        out_dict["msg1"]="Weak evidence against the null hypothesis"
        out_dict["msg2"]="Data has a unit root and is non-stationary"
        dict_kpss["msg"]="The data is non-stationary"
        second_tar_col=data["data_diff"]
        result1=adfuller(second_tar_col.dropna(),autolag='AIC')
        labels1 = ['adf_test_stat','p_value','lags_used','number_of_observations_used']
        output_1 = pd.Series(result[0:4],index=labels)
        for key,val in result1[4].items():
            l = list(key)
            print(l)
            l.remove('%')
            k = "".join(l)
            print(k)
            output_1[f'critical_value_{k}']=val
        if result1[1] < 0.05:
            output_2="stationary"
        else:
            output_2="non_stationary"
        if output_2=="stationary":
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data has become stationary"}
        else:
            final_dict={"final_message":"The ADF test was conducted again to find out whether our data has become stationary or not.The second ADF test shows that our data is non-stationary and needs to be differenciated again"}
        mdict = { "adf_test": out_dict, "kpss_test": dict_kpss, "conversion_dict": dict_2, "final_test": final_dict }
        return mdict


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


###################  TS Plot functions  ##########################

def create_corr_plot(series, plot_pacf, nlags=15):
    #import matplotlib.pyplot as plt
    # plot_pacf = False
    corr_array = sm.tsa.pacf(series.dropna(),nlags=nlags, alpha=0.05) if plot_pacf else sm.tsa.acf(series.dropna(),nlags=nlags, alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]
#     fig = make_subplots(rows=1, cols=2)
    fig = go.Figure()
    ls = [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f')
     for x in range(len(corr_array[0]))]
    print("Plot Corr X values", ls)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                   marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,nlags+2])
    fig.update_yaxes(zerolinecolor='#000000',range=[-1.2,1.2])
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'
    fig.update_layout(title=title)
    # fig.show()
    return fig
