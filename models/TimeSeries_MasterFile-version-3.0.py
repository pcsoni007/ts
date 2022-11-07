#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA,ARIMAResults

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import statsmodels.tsa as sm1
from dateutil.parser import parse
from pmdarima import auto_arima

# To do remove the graph related libraries .
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns
#import plotly.express as px
#import chart_studio.plotly as ply
#import cufflinks as cf
#import cufflinks as cf
#cf.go_offline()
#cf.set_config_file(offline=False, world_readable=True)


# In[3]:


from IPython.core.display import display, HTML
display(HTML ('<style>.container {width: 100% !important}</style'))


# In[2]:


# Importing CSV Dataset
file = r"C:\Users\plahare\Downloads\BeerWineLiquor.csv"

def file_read(data):
    return pd.read_csv(data)# data= file_path 


# In[3]:


df= file_read(file)
df


# # Setting Target col and Date col

# ### backend team, please create an object or drop _down where user can choose a target column since we are unable to call these functions in other functions.

# In[4]:


# Dropdown for selecting target column for forecasting and date column 
def target_col(data,col_name):
    target_column=data[col_name]
    #df=pd.DataFrame(target_column)
    #df.columns.names = ['Index']
    return target_column
    
    
def ts_col(data,col_name):
    time_column=data[col_name]
    #df=pd.DataFrame(time_column)
    #df.columns.names = ['Index']
    return time_column


# # Set Index

# In[7]:


# Changing date to datetime and set it as an index
def set_index(data,date_column):
    data.set_index(date_column,inplace=True)
    data.index =[ x.strip() for x in data.index]
    data.columns.names=[date_column]
    data.index=pd.to_datetime(data.index)
    
    return data

    
    
    


# # Shape of Data / Rows & Cols

# In[10]:


# displaying rows and columns #Return dictionary .eg:{"rows": 0, "cols": 0}
def shape_df(data):
    s=data.shape
    s1=s[0]
    s2=s[1]
    df={'Name':['Row_count','Column_count'],'Count':[s1,s2]}
    df1=pd.DataFrame(df)
    df1.columns.names = ['Index']
    return df1


# In[ ]:





# # Head & Tail

# In[12]:


# giving choice to user to display head or tail  
def display_head_tail1(data, choice='Head'):
    if choice == 'Head':
        df=data.head()
        return df
    elif choice=='Tail':
        df=data.tail()

        return df
    else:
        return {"message": "Invalid choice"} 
        


# In[ ]:





# # Describe function

# In[ ]:


def data_description(dataframe):
    a = dataframe.describe()
    b = a.transpose()
    d = pd.DataFrame({'col_names': pd.Series(dtype='str'),
                   'col_data_type': pd.Series(dtype='str'),
                   'col_null_val': pd.Series(dtype='int')})
    l1 =[]
    l2=[]
    l3=[]
    col = dataframe.columns
    for i in col:
        l1.append(i)
        l2.append(df[i].dtypes)
        l3.append(df[i].isnull().sum())
    d['col_names']= l1
    d['col_data_type']=l2
    d['col_null_val']=l3
    d = d.set_index("col_names", drop=True)
    d.index.name = None
    desc = d.join(b)
    desc.columns.names = ['Column_Name']
    return desc


# In[14]:



     


# ### Resampling function

# In[16]:


# exhibit frequency of data

def exhhibitFreq(dataframe, col):
    #dataframe = dataframe.reset_index()
    dataframe = dataframe.sort_values(by=col)
    dataframe[col] = pd.to_datetime(dataframe[col]) #to convert date column into datetime datatype
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


# In[ ]:


# Resampling data

def resamplingData1(dataframe, period, col):
    #dataframe = dataframe.reset_index()
    dataframe[col] = pd.to_datetime(dataframe[col])
    dataframe = dataframe.resample(period, on=col).mean().reset_index(drop=False)
    for i in dataframe.columns:
        if dataframe.dtypes[i] == int or dataframe.dtypes[i] == float:
            dataframe[i].interpolate(method='linear', limit_direction='both', inplace=True)
        else:
            dataframe[i].interpolate(method='bfill', limit_direction="backward", inplace=True)

    return dataframe.set_index(col, drop=True)


# # Null Value Treatment

# ## List of columns having null values

# In[17]:


#This functions creates a dictionary where columns are keys and values are percentage of null values present in that column 
def null_list(df):
    
    mydict={}#an empty dictionary for storing null value percentage
    list1=[]
    for i in df.columns:
        if df[i].isnull().sum()>0: #this is to create a dictionary with columns which has null values.
            mydict[i]=[(df.isnull().sum())*100 / len(df)][0][i]
    for j,k in mydict.items():
        list1.append(j)
    
    if len(list1)==0: 
        Message={"Message": "This dataset doesn't have any null values , kindly proceed with the EDA "} 
        
        return Message
    else:
        return mydict


# In[ ]:





# ## Graph to display percentage of null values

# In[19]:


#for plotting the null values. this function plots graph of columns in the x-axis and its percentage of null values in the y-axis
def graph(data):
    mydict={}#an empty dictionary for storing the null values and its percentage
    for i in data.columns:
        mydict[i]=[(data.isnull().sum())*100 / len(data)][0][i]
    null_percentage=mydict.values()
    x=(np.array(data.columns)).tolist()
    y=list(null_percentage)
    my_dict={"x_label": 'Columns',
             "y_label":'Pecentages',
             "title": "Percentage of null values present in each column",
             "x_value":x,
             "y_value":y,
             "chart_type":'bar'}
    return my_dict  


# In[ ]:





# ## Null Value Treatment

# In[ ]:





# In[ ]:





# In[22]:



############################### Above function Edited by Vijay#############################################
def impute(df, col_name, impute_method='interpolation'):
    if df.dtypes[col_name] == str or df.dtypes[col_name] == object:
        df[col_name].fillna(df[col_name].mode()[0], inplace=True)
    else:
        flag1 = (df[col_name].isnull() & df[col_name].shift(-1).isnull()).any()
        flag2 = df[col_name].head(1).isnull().bool()
        flag3 = df[col_name].tail(1).isnull().bool()
        if flag1 or flag2 or flag3:
            df[col_name].fillna(df[col_name].interpolate(method='linear', limit_direction="both"), inplace=True)
        elif impute_method == "locf" and (flag1 == False and flag2 == False and flag3 == False):
            df[col_name].fillna(df[col_name].ffill(),inplace=True)
        elif impute_method == "nocb" and (flag1 == False and flag2 == False and flag3 == False):
            df[col_name].fillna(df[col_name].bfill(),inplace=True)


# In[23]:


def drop_rows(df, col_name):
    df.dropna(subset=[col_name], axis=0, how="any", inplace=True)


# In[24]:


def drop_cols(df, col_name):
    df.drop([col_name],axis=1,inplace=True)


# # EDA

# ## Date vs target_col

# In[21]:


def plot_col(df,target_col):
    Dict={"Interpretation" : "This graph represents visualization of dependent or target variable w.r.t Time.This depicts how the dependent variable varies with the time. X axis represents time and Y axis represents dependent variable. "}
   
    my_dict={"x_label": 'Time',
             "y_label":'Target column values',
             "title": "Target variable w.r.t. time",
             "x_value":df.index,
             "y_value":df[target_col],
             "chart_type":'lineplot'}
    return my_dict, Dict
   


# In[ ]:





# ## Resampled plot

# In[24]:


#This function is from UI perspective
def resample_plot(data,target_col,resample_alias="M"):
    A=pd.DataFrame(data[target_col].resample(resample_alias).max())
    my_dict={
        "title":"Resampled Graph Of Target Variable",
        "x_label":'Date',
        "y_label": 'Target Column',
        "x_values":A.index,
        "y_values":A.values,
        "Chart_type":"BarPlot"}
    return my_dict


# ## Top n Values

# In[26]:


# displaying top n values in dataframe
def top_n_values(data,target_col,n=10):
    #n = int(input("How many top values do you want to see?\n"))
    #My_dict={"Below are the top {0} values in the {1} column": ".format(n, col_name)} 
    top_values=pd.DataFrame(data[target_col].sort_values(ascending = False).head(10))
    top_values.columns.names = ['date']
    return top_values


# In[28]:


def plot_top_n(data,target_col):
    dict1={"Interpretation":"\n This graph represents visualization of Top values of dependent or target variable w.r.t Time. X axis represents time and Y axis represents top values of dependent variable. "}
    top_n_dataf =top_n_values(data,target_col,n=10)
    my_dict={
        "title":"Visualization of Top N values Of Target Variable",
        "x_label":'Date',
        "y_label": 'Target Column',
        "x_values": data.index,
        "y_values":top_n_dataf.values,
        "Chart_type":"BarPlot"}
    return my_dict,dict1


# # Stationarity Check

# ## Seasonal Decompose during EDA

# In[30]:


# seasonal decomposition plot
# seasonal decomposition plot
def decomposition(target_col,choice='A'):
    y = target_col.to_frame()
    if choice == 'M':    
    # Multiplicative Decomposition
        result=seasonal_decompose(y, model='multiplicative')       
        Date= y.index
        
        observed = result.observed
        Trend_comp= result.trend
        Seasonal_comp= result.seasonal
        Residual_comp= result.resid
        
        mydict_Observed={
        "title":"Multiplicative Decompose",
        "x_label":'Date',
        "y_label": 'Observed',
        "x_values": Date,
        "y_values":observed.values,
        "Chart_type":"LineChart"}
#         return mydict_Observed
    
    
        mydict_Trend={
        "x_label":'Date',
        "y_label": 'Trend',
        "x_values": Date,
        "y_values":Trend_comp.values,
        "Chart_type":"LineChart"}
#         return mydict_Trend
      
    
        mydict_seasonal={
        "x_label":'Date',
        "y_label": 'Seasonal',
        "x_values": Date,
        "y_values":Seasonal_comp.values,
        "Chart_type":"LineChart"}
#         return mydict_seasonal
    
        mydict_Resid={
        "x_label":'Date',
        "y_label": 'Residual',
        "x_values": Date,
        "y_values":Residual_comp.values,
        "Chart_type":"LineChart"}
        
        My_dict={"Interpretation" : "Here X axis represents Time and Y axis represents Normal scaled data. Time series has 4 components Trend,seasonality,cyclical variation and irregular variation.",

                  "Trend component": "This is useful in predicting future movements. Over a long period of time, the trend shows whether the data tends to increase or decrease",

                  "Seasonal component": "The seasonal component of a time series is the variation in some variable due to some predetermined patterns in its behavior.",

                  "Cyclical component": "The cyclical component in a time series is the part of the movement in the variable which can be explained by other cyclical movements in the economy.",

                  "Irregular component": "this term gives information about non-seasonal patterns.",

                  "Note":"Time series has two types of decomposition models Additive Model and Multiplicative model. The plot shows the decomposition of your time series data in its seasonal component, its trend component and the remainder. If you add or multiply the decomposition together you would get back the actual data. First block represents original series , second represents trend , third represents seasonality presents, fourth represents error component or residual.For additive if we add below three blocks we get original data series. Similarly for multiplicative we have to multiply the components."}

        return mydict_Observed,mydict_Trend,mydict_seasonal,mydict_Resid,My_dict

    elif choice == 'A':
        
    # Additive Decomposition
        result=seasonal_decompose(y, model='additive')
        
        Date= y.index
        observed = result.observed
        Trend_comp= result.trend
        Seasonal_comp= result.seasonal
        Residual_comp= result.resid

        mydict_Observed={
        "title":"Additive Decompose",
        "x_label":'Date',
        "y_label": 'Observed',
        "x_values": Date,
        "y_values":observed.values,
        "Chart_type":"LineChart"}
#        return mydict_Observed
    
    
        mydict_Trend={
        "x_label":'Date',
        "y_label": 'Trend',
        "x_values": Date,
        "y_values":Trend_comp.values,
        "Chart_type":"LineChart"}
#        return mydict_Trend
    
        mydict_seasonal={
        "x_label":'Date',
        "y_label": 'Seasonal',
        "x_values": Date,
        "y_values":Seasonal_comp.values,
        "Chart_type":"LineChart"}
#        return mydict_seasonal
    
        mydict_Resid={
        "x_label":'Date',
        "y_label": 'Residual',
        "x_values": Date,
        "y_values":Residual_comp.values,
        "Chart_type":"LineChart"}
        

        My_dict={"Interpretation" : "Here X axis represents Time and Y axis represents Normal scaled data. Time series has 4 components Trend,seasonality,cyclical variation and irregular variation.",

                 "Trend component": "This is useful in predicting future movements. Over a long period of time, the trend shows whether the data tends to increase or decrease",

                "Seasonal component": "The seasonal component of a time series is the variation in some variable due to some predetermined patterns in its behavior.",
               
                 "Cyclical component": "The cyclical component in a time series is the part of the movement in the variable which can be explained by other cyclical movements in the economy.",
                
                 "irregular component": "this term gives information about non-seasonal patterns.",

                 "Note":"Time series has two types of decomposition models Additive Model and Multiplicative model. The plot shows the decomposition of your time series data in its seasonal component, its trend component and the remainder. If you add or multiply the decomposition together you would get back the actual data. First block represents original series , second represents trend , third represents seasonality presents, fourth represents error component or residual.For additive if we add below three blocks we get original data series. Similarly for multiplicative we have to multiply the components."}

        return mydict_Observed,mydict_Trend,mydict_seasonal,mydict_Resid,My_dict
    


# ## Stationarity Check

# ## Stationarity Check Plot

# In[32]:


# Plot for checking stationarity
def stationarity_check_plot(timeseries,target_col):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    df_x= df.index
    df_y= df[target_col]

    x_rollmean=rolmean.index
    y_rollmean=rolmean[target_col]
    
    x_rolstd=rolstd.index
    y_rolstd=rolstd[target_col]
    
    
    mydict_df={ 
        "title":"Stationarity check Plot",
        "x_label" :'Date',
        "y_label" :'Target',
        "legends" :['Original'],
        "x_values":df_x,
        "y_values":df_y,
        "Chart_type":"StackedLineChart"
    }
   
    mydict_rollmean={ 
        "x_label":'Date',
        "y_label": 'RollingMean',
        "legends":['RollingMean'],
        "x_values": x_rollmean,
        "y_values":y_rollmean,
        "Chart_type":"StackedLineChart"}

    mydict_rollstd={
        "x_label":'Date',
        "y_label": 'RollingMean',
        "legends":['RollingMean'],
        "x_values": x_rollmean,
        "y_values":y_rollmean,
        "Chart_type":"StackedLineChart"}
    
    My_Dict={"Interpretation": " ",
             
             "Stationarity": " Stationarity means that the statistical properties of a process generating a time series do not change over time. That is Mean and Standard deviation is approximately constant over time.\n\nStationarity Graph represents stationarity of the series w.r.t. Time. X axis depicts time and Y axis depicts Dependent variable . Blue line represents the original Time series data , Red line represents Mean of the series data and Black line represents standard deviation of the series. "}
    
    return mydict_df,mydict_rollmean,mydict_rollstd,My_Dict
    


# ### Stationarity test- adf , kpss , and conversion from stationary to non-stationary

# In[ ]:


def stationarity_test3(data,series):
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data  
    labels = ['ADF test statistic','p-value','# lags used','number_of_observations_used']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    if result[1] < 0.05:        
        out1="stationary"
    else:
        out1="non_stationary"
        
    statistic, p_value, n_lags, critical_values = kpss(series)
    dict_kpss={"KPSS test statistic":statistic,"p_value":p_value,"n_lags":n_lags,"critical_values":critical_values}
    #kpss_result=pd.DataFrame(dict_kpss.items(), columns=['KPSS test statistic', 'p_value','n_lags','critical_values'])
    # Format Output
    
    if p_value<0.05:
        out2="stationary"
    else:
        out2="non_stationary"
    if out1=="stationary" and out2=="stationary":
        dict_1={"Message":"Since both ADF and KPSS test results indicates stationarity,the data is stationary. Kindly proceed further "}
        out_dict=out.to_dict()
        out_dict["Message1"]="Strong evidence against the null hypothesis"
        out_dict["Message2"]="Data has no unit root and is stationary"
        dict_kpss["Message11"]="The data is stationary"
        return out_dict,dict_kpss,dict_1
        
    elif out1=="stationary" and out2=="non_stationary":
        data["diff_1"] =series.diff(periods=1)
        data['diff_1'].dropna()
        dict_2={"Message":"KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary.Therefore differencing has been used to make the data stationary."}
        out_dict=out.to_dict()
        out_dict["Message1"]="Strong evidence against the null hypothesis"
        out_dict["Message2"]="Data has no unit root and is stationary"
        
        dict_kpss["Message11"]="The data is non-stationary"
        result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data  
        labels = ['ADF test statistic','p-value','# lags used','number_of_observations_used']
        out = pd.Series(result[0:4],index=labels)
        for key,val in result[4].items():
            out[f'critical value ({key})']=val
        
        return out_dict,dict_kpss,dict_2
        
    elif out1=="non_stationary" and out2=="stationary":
        
        data['data_log']=np.sqrt(series)
        data['data_diff']=data['data_log'].diff().dropna()
        dict_2={"Message":"KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary.Therefore log transformation has been used to make the data strict stationary."}
        out_dict=out.to_dict()
        out_dict["Message1"]="Weak evidence against the null hypothesis"
        out_dict["Message2"]="Data has a unit root and is non-stationary"
        dict_kpss["Message11"]="The data is stationary"
        return out_dict,dict_kpss,dict_2
    elif out1=="non_stationary" and out2=="non_stationary":
        data['data_log']=np.sqrt(series)
        data['data_diff']=data['data_log'].diff().dropna()
        dict_2={"Message":"Both ADF and KPSS test indicates non-stationarity.Therefore log transformation has been chosen to make the data stationarity"}
        out_dict=out.to_dict()
        out_dict["Message1"]="Weak evidence against the null hypothesis"
        out_dict["Message2"]="Data has a unit root and is non-stationary"
        dict_kpss["Message11"]="The data is non-stationary"
        return out_dict,dict_kpss,dict_2
       


# In[33]:


# ADF test
def adf_test(target_col):

    result = adfuller(target_col.dropna(),autolag='AIC') # .dropna() handles differenced data  
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    if result[1] < 0.05:        
        Dict_1={"message1":"Strong evidence against the null hypothesis"}
        Dict_2={"message2":"Reject the null hypothesis"}
        Dict_3={"message3":"Data has no unit root and is stationary"}
    else:
        Dict_1={"message1":"Weak evidence against the null hypothesis"}
        Dict_2={"message2":"Fail to reject the null hypothesis"}
        Dict_3={"message3":"Data has a unit root and is non-stationary"}

    return out, Dict_1,Dict_2,Dict_3 


# In[35]:


# KPSS test for stationarity and display output
from statsmodels.tsa.stattools import kpss
def kpss_test(target_col):  
    statistic, p_value, n_lags, critical_values = kpss(target_col)
    # Format Output
    Dict_1={'KPSS Statistic': statistic}
    Dict_2={'p-value': p_value}
    Dict_3={'num lags': n_lags}
    Dict_4={'Critial Values':critical_values}
    

    return Dict_1,Dict_2,Dict_3,Dict_4,(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


# In[37]:


# Conversion of non stationarity to stationarity
def non_stationarity_stationarity(data,target_col,adf,kpss,choice='T'):    
    if adf=="stationary" and kpss=="stationary":        
        Dict_1={"Message":"Data has no unit root and is Stationary"} 
        return Dict_1        
    elif adf=="non stationary" and kpss=="non stationary":        
        Dict_2={"Message":"The data has been converted into Stationary data successfully!!!"}
        if choice == 'T':  
            data['data_log']=np.sqrt(target_col)
            data['data_diff']=data['data_log'].diff().dropna()
            return Dict_2,adf_test(data['data_diff']); 
        
        elif choice == 'D':
            data["diff_1"] =target_col.diff(periods=1)
            data['diff_1'].dropna()
            return Dict_2,adf_test(data['diff_1']); 
              
    elif adf=="non stationary" and kpss=="stationary":
        data['data_log']=np.sqrt(target_col)
        data['data_diff']=data['data_log'].diff().dropna()
        return Dict_2,adf_test(data['data_diff']); 
        
    elif adf=="stationary" and kpss=="non stationary":
        
        data["diff_1"] =target_col.diff(periods=1)
        data['diff_1'].dropna()
        return Dict_2,adf_test(data['diff_1']);         
    else:
        Dict_3={"Warning": "Please enter valid input"}
        return Dict_3


# # ACF PACF
# 

# In[ ]:


#this acf_pacf function 
def acf_pacf(series,nlags=15,alpha=0.05):
    acf, ci = sm.tsa.acf(df1,nlags=15, alpha=0.05)
    pacf, ci1 = sm.tsa.pacf(df1,nlags=15, alpha=0.05)
    mydict_acf={
        'title':'ACF_plot',
        'x_label':"Lags"
        'y_label':"PACF_values"
        'x_val':[i for i in range(0,nlags+2)],
        'y_val_acf':acf, #y values have two components acf values and confidence interval values.
        'y_val_confidence_interval':ci,
        'type':'stem_plot-aka lollipop plots'}
    mydict_pacf={ 
        "title":"PACF Plot",
        "x_label" :'Lags',
        "y_label" :'PACF_values',
        "x_values":[j for j in range(0,nlags+2)],
        "y_values":pacf,
        "y_val_confidence_interval":ci1
        "Chart_type":"stem_plot-aka lollipop plots"}
    
    dict1={"Interpretation":"ACF represnts auto correlation between varibles w.r.t Time into consideration all components of time series.PACF represnts correlation function of the variables with residuals partially."}
    dict2={"":"Both ACF & PACF starts at lag 0 , which is the correlation of variables with itself and therefore results in a correlation of 1. Difference between both is inclusion and exclusion of indirect correlations. Blue area depicts 95% confidence interval."}
    dict3={ "Sharp Drop Point": 
            ["Instant drop lag just after lag 0.",

            "ACF sharp drop point implies MA order & PACF sharp drop point implies AR order",

            "Some basic approach for model choosing are as follows",

            "1. ACF plot declines gradually and PACF drops instantly use AR model.",
            "2. ACF drops instantly and PACF declines gradually use MA model.", 
            "3. Both declines gradually use ARMA model",
            "4. Both drops instantly we are not able to model the time series."]}

    dict4={"Note":

            "ARIMA and SARIMA models are Intergrated ARMA models we will use the same identified orders from both the plots."}
    return mydict_acf,mydict_pacf,dict1,dict2,dict3,dict4
    
    


# # Train Test Split

# In[ ]:


# spliting dataset
def split(data,size_input=0.75):
    #size_input=float(input("Please enter the size of percentage where you want to split the data-for eg 0.75 for 75% or 0.80 for 80%"))
    #splitting 85%/15% because of little amount of data
    size = int(len(data) * size_input)
    train= data[:size]
    test = data[size:]
    return(train,test)
    
    
    



# In[ ]:


train,test= split(df)


# In[ ]:


train.shape


# In[ ]:


test.shape


# # Forecasting 

# ## AutoArima

# In[2]:


# Autoarima model
def gen_auto_arima(df, col, m, f, periods, maxp=5, maxd=2, maxq=5, maxP=5, maxD=2, maxQ=5):
    automodel= auto_arima(df[col], seasonal=True, m=m, start_p=0, start_q=0, d=None, D=None, stepwise=True, max_p= maxp, max_d= maxd, max_q = maxq,
                         max_P= maxP, max_D= maxD, max_Q= maxQ)
    #print(automodel.summary())
    preds, confint = automodel.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1], periods = periods, freq=f)
    fitted_series = pd.DataFrame(preds, index=index_of_fc)
    
    #print(preds)
    
    mydict={ 
        "title":"Auto ARIMA Model",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Auto_ARIMA_Forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":fitted_series, # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col],fitted_series)).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-fitted_series)/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Auto ARIMA Model'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    return mydict,results


# ### Simpler timeseries models

# In[1]:


def naive_method(test_df):
    y_hat_naive = test_df.copy()
    y_hat_naive['naive_forecast'] = train[target_col][train_len-1]
    
    mydict={ 
        "title":"Naive Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Naive_Forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_naive['naive_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_naive['naive_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_naive['naive_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    return mydict,results


# In[2]:


def average_method(test_df):
    y_hat_average = test_df.copy()
    y_hat_average['average_forecast'] = train[target_col].mean()
    
    mydict={ 
        "title":"Average Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Average_Forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_average['average_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_average['average_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_average['average_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Average method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    return mydict,results
    


# In[3]:


def simple_moving_average(df, ma_window):
    y_hat_sma = df.copy()
    y_hat_sma['sma_forecast'] = data[target_col].rolling(ma_window).mean()
    y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]
    
    mydict={ 
        "title":"Simple Moving Average Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Simple_Moving_Average_Forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_sma['sma_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_sma['sma_forecast'][train_len:])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_sma['sma_forecast'][train_len:])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })
    results = results[['Method', 'RMSE', 'MAPE']]
    return mydict,results


# In[4]:


def simple_exponential_smoothing(test_df,forecast_duration):
    model = SimpleExpSmoothing(train[target_col])
    model_fit = model.fit(smoothing_level=0.2,optimized=False)
    model_fit.params
    y_hat_ses = test_df.copy()
    y_hat_ses['ses_forecast'] = model_fit.forecast(forecast_duration)
    
    mydict={ 
        "title":"Simple Exponential Smoothing Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Simple exponential smoothing forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_ses['ses_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_ses['ses_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_ses['ses_forecast'])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results


# In[5]:


def HoltExponentialSmoothing(train_df,test_df,seasonal_periods,forecast_duration):
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='additive', seasonal=None)
    model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)
    print(model_fit.params)
    y_hat_holt = test_df.copy()
    y_hat_holt['holt_forecast'] = model_fit.forecast(forecast_duration)
    
    mydict={ 
        "title":"Holt\'s Exponential Smoothing Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Holt\'s exponential smoothing forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_holt['holt_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_holt['holt_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_holt['holt_forecast'])/test['Passengers'])*100,2)

    results = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results


# In[6]:


def Holtwinter_exponentialsmoothing_additive(train_df,test_df,seasonal_periods,forecast_duration):    
    y_hat_hwa = test.copy()
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='add', seasonal='add')
    model_fit = model.fit(optimized=True)
    y_hat_hwa['hw_forecast'] = model_fit.forecast(forecast_duration)
    
    mydict={ 
        "title":"Holt Winters\' Additive Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Holt Winters\'s additive forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_hwa['hw_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_hwa['hw_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_hwa['hw_forecast'])/test['Passengers'])*100,2)
    results = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results


# In[7]:


def Holtwinter_exponentialsmoothing_multiplicative(train_df,test_df,seasonal_periods,forecast_duration):    
    y_hat_hwa = test.copy()
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='add', seasonal='mul')
    model_fit = model.fit(optimized=True)
    y_hat_hwa['hw_forecast'] = model_fit.forecast(forecast_duration)
    
    mydict={ 
        "title":"Holt Winters\' multiplicative Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Holt Winters\'s multiplicative forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_hwa['hw_forecast'], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_hwa['hw_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_hwa['hw_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results


# In[8]:


def autoregressive_function(data,train_df,test_df):
    data_boxcox = pd.Series(boxcox(data[target_col], lmbda=0), index = data.index)
    data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), data.index)
    train_data_boxcox = data_boxcox[:train_len]
    test_data_boxcox = data_boxcox[train_len:]
    train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
    test_data_boxcox_diff = data_boxcox_diff[train_len-1:]
    model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) 
    model_fit = model.fit()
    y_hat_ar = data_boxcox_diff.copy()
    y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
    y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()
    y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[2])
    y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])
    
    mydict={ 
        "title":"Auto Regression Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Auto regression forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_ar['ar_forecast'][test.index.min():], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_ar['ar_forecast'][test.index.min():])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_ar['ar_forecast'][test.index.min():])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Autoregressive (AR) method'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results
    


# In[9]:


def moving_average_function(data,train_df,test_df):
    data_boxcox = pd.Series(boxcox(data[target_col], lmbda=0), index = data.index)
    data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), data.index)
    train_data_boxcox = data_boxcox[:train_len]
    test_data_boxcox = data_boxcox[train_len:]
    train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
    test_data_boxcox_diff = data_boxcox_diff[train_len-1:]
    model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) 
    model_fit = model.fit()
    y_hat_ar = data_boxcox_diff.copy()
    y_hat_ar['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
    y_hat_ar['ma_forecast_boxcox'] = y_hat_ar['ma_forecast_boxcox_diff'].cumsum()
    y_hat_ar['ma_forecast_boxcox'] = y_hat_ar['ma_forecast_boxcox'].add(data_boxcox[2])
    y_hat_ar['ma_forecast'] = np.exp(y_hat_ar['ma_forecast_boxcox'])
    
    mydict={ 
        "title":"Moving Average Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Moving Average forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_ar['ma_forecast'][test.index.min():], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_ar['ma_forecast'][test.index.min():])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_ar['ma_forecast'][test.index.min():])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Moving Average (MA) method'], 'RMSE': [rmse],'MAPE': [mape] })
    return results


# In[10]:


def autoregressive_moving_average_function(data,train_df,test_df):
    data_boxcox = pd.Series(boxcox(data[target_col], lmbda=0), index = data.index)
    data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), data.index)
    train_data_boxcox = data_boxcox[:train_len]
    test_data_boxcox = data_boxcox[train_len:]
    train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
    test_data_boxcox_diff = data_boxcox_diff[train_len-1:]
    model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1)) 
    model_fit = model.fit()
    y_hat_ar = data_boxcox_diff.copy()
    y_hat_ar['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
    y_hat_ar['arma_forecast_boxcox'] = y_hat_ar['arma_forecast_boxcox_diff'].cumsum()
    y_hat_ar['arma_forecast_boxcox'] = y_hat_ar['arma_forecast_boxcox'].add(data_boxcox[2])
    y_hat_ar['arma_forecast'] = np.exp(y_hat_ar['arma_forecast_boxcox'])
    
    mydict={ 
        "title":"Autoregressive Moving Average Method",
        "x_label" :'Date',
        "y_label" :'Values',
        "legends":['Train_data','Test_data','Auto regression moving average forecast'],
        "x_values":Date,
        "y1_values":train[target_col], # upto this line in blue colour
        "y2_values":test[target_col], # this line in orange colour
        "y3_values":y_hat_ar['arma_forecast'][test.index.min():], # this line in green colour
        "Chart_type":"line plot"}
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_ar['arma_forecast'][test.index.min():])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_ar['arma_forecast'][test.index.min():])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Autoregressive Moving Average (ARMA) method'], 'RMSE': [rmse],'MAPE': [mape] })
    return mydict,results


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




