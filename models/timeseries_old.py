# %%
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


# %% [markdown]
# Need output in form of dataframe, dictinary or json for graph related functions 

# %%
# Importing CSV Dataset
file = r"C:\Users\plahare\Downloads\BeerWineLiquor.csv"
#create function that takes a file for reading and creating dataframe
def file_read(data):
    return pd.read_csv(data)# data= file_path 

# %% [markdown]
# # Setting Target col and Date col

# %% [markdown]
# create functions for target column and index column

# %%
# Dropdown for selecting target column for forecasting and date column 
def target_col(col_name):
    target_column=data[col_name]
    df=pd.DataFrame(target_column)
    df.columns.names = ['Index']
    return df
    
    
def ts_col(col_name):
    time_column=data[col_name]
    df=pd.DataFrame(time_column)
    df.columns.names = ['Index']
    return df
# Drop down

# %% [markdown]
# # Set Index

# %%
# Changing date to datetime and set it as an index
def set_index(data,col_name=ts_col("date")):
    data[col_name] = pd.to_datetime(data[col_name])
    df=data.set_index(col_name,inplace=True)
    df.columns.names = ['Index']
    #print(data.head())
    return df
    
    

# %% [markdown]
# # Shape of Data / Rows & Cols

# %%
# displaying rows and columns #Return dictionary .eg:{"rows": 0, "cols": 0}
def shape_df(data):
    s=data.shape
    s1=s[0]
    s2=s[1]
    df={'Name':['Row_count','Column_count'],'Count':[s1,s2]}
    df1=pd.DataFrame(df)
    df1.columns.names = ['Index']
    return df1
    #use return here to print dict
    #print('No of rows :{}'.format(s[0]))
    #print('No of Columns:{}'.format(s[1]))

# %% [markdown]
# # Head & Tail

# %%
# giving choice to user to display head or tail  
def display_head_tail1(data, choice='Head'):
    if choice == 'Head':
        df=data.head()
        df.columns.names = ['Index']
        return df
    elif choice=='Tail':
        df=data.tail()
        df.columns.names = ['Index']
        return df
    else:
        return {"message": "Invalid choice"} #convert this into dataframe and display
        

# %% [markdown]
# # Describe function

# %%
def describe_data2(data):
    df=data.describe().transpose()
    df.columns.names = ['Column']
    return df
     

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
#check_Continuity(df)

# %%


# %% [markdown]
# # Null Value Treatment

# %% [markdown]
# ## List of columns having null values

# %%
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
        print("This dataset doesn't have any null values , kindly proceed with the EDA .\n") #this print statment in terms of dict or dataframe
    else:
        return mydict# the output is in dictionary form 

# %% [markdown]
# ## prefer dataframe first then Dictionary then at last json

# %% [markdown]
# ## Graph to display percentage of null values

# %%
#for plotting the null values. this function plots graph of columns in the x-axis and its percentage of null values in the y-axis
def graph(data):
    
    null_percentage=pd.DataFrame(data.isnull().sum()*100)/len(data)
    #x=[data.columns]#convert this into list
    #y=[null_percentage]# convert this into list
    x=(np.array(data.columns)).tolist()
    #print(x)
    y=(np.array(null_percentage)).tolist()
    #print(y)
    print(null_percentage)
    

  
    my_dict={"x_label": 'Columns',
             "y_label":'Pecentages',
             "title": "Percentage of null values present in each column",
             "x_value":x,
             "y_value":y,
             "chart_type":'bar'}
    return my_dict



# %% [markdown]
# ## Null Value Treatment

# %%


# %%
#This function iterates through whole dataframe and treats the missing values with appropriate method chosen by the user.
#Vijay will look into it.
#generate two lists - columns with null values and columns with non null values-suggestion
#create where user can give multiple inputs.-suggestion
#need an alternative
def null_values(df):
    mydict={}#an empty dictionary for storing the null values and its percentage
    for i in df.columns:
        mydict[i]=[(df.isnull().sum())*100 / len(df)][0][i]
    print(mydict)
    #looping through the whole dataframe using dictionary"mydict"
    for i,k in mydict.items():
        print(i)
        #if k=0 then it will move on to the next column
        if k==0:
            pass
        else:
            #if k>0 then it will ask the user to treat the column accordingly 
            flag=True
            while flag:
                #declare choice within function parameter
                choice=input("Kindly choose whether you want to opt for dropping the rows/columns or would like to impute the values? Please type 'drop_rows' for dropping the rows or 'drop_column' for dropping the columns and 'impute' for filling the missing values\n")
                if choice=="drop_rows" or choice=="drop_column" or choice=="impute":
                    flag=False
                else :
                    print("enter a valid choice")
            #if user chooses to drop the rows, then it will perform the following operation
            if choice=="drop_rows":
                df.dropna(subset=[i],axis=0,how="any",inplace=True)#user should provide how
                mydict1={}
                for i in df.columns:
                    mydict1[i]=[(df.isnull().sum())*100 / len(df)][0][i]
                mydict.update(mydict1)
             #if user chooses to drop the column, then it will perform the following operation   
            elif choice=="drop_column":
                df.drop([i],axis=1,inplace=True)
            #if user chooses to impute the missing values, then it will perform the following operation
            elif choice=="impute":
                if df.dtypes[i]==str or df.dtypes[i]==object:
                    df[i].fillna(df[i].mode()[0], inplace=True)
                else:
                    
                    boolean=(df[i].isnull() & df[i].shift(-1).isnull()).any()
                    boolean1=df[i].head(1).isnull().bool()
                    boolean2=df[i].tail(1).isnull().bool()
                    if boolean==True or boolean1==True or boolean2==True:
                        df[i].fillna(df[i].interpolate(method='linear',limit_direction="both"),inplace=True)
                    else:
                        Flag1=True
                        while Flag1:
                        
                            impute=input("Kindly Choose any one method for imputing missing values - please type 'LOCF' or 'NOCB'  or 'Interpolation'.\n")
                            if impute=="LOCF" or impute=="NOCB" or impute=="Interpolation":
                                Flag1=False
                            else:
                                print("enter a valid input")
                        
                        if impute=="LOCF":
                            df[i].fillna(df[i].ffill(),inplace=True)
                        elif impute=="NOCB":
                            df[i].fillna(df[i].bfill(),inplace=True)
                        elif impute=="Interpolation":
                            df[i].fillna(df[i].interpolate(method='linear',limit_direction="both"),inplace=True)
                        
                            
            
                
    print(df.isnull().sum())
    print("The null values have been successfully treated!")            

# %% [markdown]
# # EDA

# %% [markdown]
# ## Date vs target_col

# %%
def plot_col(df,col_name):
    print("Interpretation:\n This graph represents visualization of dependent or target variable w.r.t Time.This depicts how the dependent variable varies with the time. X axis represents time and Y axis represents dependent variable. ")
   
    my_dict={"x_label": 'Time',
             "y_label":'Target column values',
             "title": "Target variable w.r.t. time",
             "x_value":df.index,
             "y_value":df[target_col],
             "chart_type":'lineplot'}
    return my_dict
     #df.plot(ts_col,col_name,figsize=(12,6),title=title).autoscale(axis='both',tight=True);

# %%
plot_col(df,target_col)#check what is the error #error corrected

# %%


# %%
#code to show the alias image to the user
from IPython.display import Image
Image(filename="C:\\Users\\DB4\\Downloads\\MicrosoftTeams-image.png",width=1000,height=400)

# %% [markdown]
# ## Resampled plot

# %%
#This function is from UI perspective
def resample_plot(data,col_name,resample_alias="M"):
    data[col_name].resample(resample_alias).max()# resample aliaces ask to vijay
    my_dict={
        "title":"Resampled Graph Of Target Variable",
        "x_label":'Date',
        "y_label": 'Target Column',
        "x_values": '', # yet to give 
        "y_values": '', # yet to decide
        "Chart_type":"BarPlot"}
    return my_dict

# %% [markdown]
# ## Top n Values

# %%
# displaying top n values in dataframe
def top_n_values(data,col_name,n=10):
    #n = int(input("How many top values do you want to see?\n"))
    print("Below are the top {0} values in the {1} column: ".format(n, col_name))
    return pd.DataFrame(data[col_name].sort_values(ascending = False).head(10))

# %%
def plot_top_n(data, col_name):
    print("Interpretation:\n This graph represents visualization of Top values of dependent or target variable w.r.t Time. X axis represents time and Y axis represents top values of dependent variable. ")
    my_dict={
        "title":"Visualization of Top N values Of Target Variable",
        "x_label":'Date',
        "y_label": 'Target Column',
        "x_values": data.index,
        "y_values": top_n_dataf.values,
        "Chart_type": "BarPlot"}
    return my_dict

# %% [markdown]
# # Updated functions for pre processing and EDA is as above . Below functions are yet to be discussed with Vijay

# %% [markdown]
# # Stationarity Check

# %% [markdown]
# ## Seasonal Decompose during EDA

# %%
# seasonal decomposition plot
def decomposition(series, choice):
    
    message = """Interpretation:\n Here X axis represents Time and Y axis represents Normal scaled data. Time series has 4 components Trend,seasonality,cyclical variation and irregular variation. \n Trend component: This is useful in predicting future movements. Over a long period of time, the trend shows whether the data tends to increase or decrease. \n 
            Seasonal component: The seasonal component of a time series is the variation in some variable due to some predetermined patterns in its behavior. \n Cyclical component: The cyclical component in a time series is the part of the movement in the variable which can be explained by other cyclical movements in the economy. \n  irregular component: this term gives information about non-seasonal patterns.\n
            \nTime series has two types of decomposition models Additive Model and Multiplicative model. The plot shows the decomposition of your time series data in its seasonal component, its trend component and the remainder. If you add or multiply the decomposition together you would get back the actual data. First block represents original series , second represents trend , third represents seasonality presents, fourth represents error component or residual. 
            \nFor additive if we add below three blocks we get original data series. Similarly for multiplicative we have to multiply the components."""
    
    y = series.to_frame()
    
    if choice == 'M':
       return seasonal_decompose(y, model='multiplicative',period = 52), message

    elif choice == 'A':
       return seasonal_decompose(y, model='additive',period = 52), message

    elif choice=="MA":
        return seasonal_decompose(y, model='multiplicative',period = 52),  seasonal_decompose(y, model='additive',period = 52), message

    
    return ({"message": "This is invalid choice. Please choose Either M or A"})
    
    


# %%
decomposition(df[target_col])

# %% [markdown]
# ## Stationarity Check

# %% [markdown]
# ## Stationarity Check Plot

# %%
# Plot for checking stationarity
def stationarity_check_plot(timeseries, col_name):

    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(20,6))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation for {}'.format(col_name))
    plt.show(block=False)
    print("Interpretation:\n\n Stationarity:\n\n Stationarity means that the statistical properties of a process generating a time series do not change over time. That is Mean and Standard deviation is approximately constant over time.\n\nStationarity Graph represents stationarity of the series w.r.t. Time. X axis depicts time and Y axis depicts Dependent variable . Blue line represents the original Time series data , Red line represents Mean of the series data and Black line represents standard deviation of the series. ")

# %%
stationarity_check_plot(df,target_col)

# %%
# Adf test for checking stationarity and display output to user
def adf_test(series):
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
   
    print(f'Augmented Dickey-Fuller Test: ')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string(), '\n')          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
    return out


# %%
adf_test(df[target_col])

# %%
# KPSS test for stationarity and display output
from statsmodels.tsa.stattools import kpss
def kpss_test(series):  
    statistic, p_value, n_lags, critical_values = kpss(series)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

# %%
kpss_test(df)

# %%
# Conversion of non stationarity to stationarity
def non_stationarity_stationarity(data,series,adf,kpss):
    #adf=input("Enter result of adf test either stationary or non stationary :\n")
    #kpss=input("Enter result of kpss test either stationary or non stationary :\n")
    
    if adf=="stationary" and kpss=="stationary":
        
        print("Data has no unit root and is Stationary")
        
    elif adf=="non stationary" and kpss=="non stationary":
        
        print("Data has unit root and is non stationary, please make data stationary")
        
        choice=input("Enter T for Transformation method or D for differencing method :\n")
    
        if choice == 'T':
            data['data_log']=np.sqrt(series)
            data['data_diff']=data['data_log'].diff().dropna()
            adf_test(data['data_diff']); 
        
        elif choice == 'D':
            data["diff_1"] =series.diff(periods=1)
            data['diff_1'].dropna()
            adf_test(data['diff_1']); 
        
        else:
            print(" This is invalid choice. Please choose Either T or D")
        
    elif adf=="non stationary" and kpss=="stationary":
        data['data_log']=np.sqrt(series)
        data['data_diff']=data['data_log'].diff().dropna()
        adf_test(data['data_diff']); 
        
    elif adf=="stationary" and kpss=="non stationary":
        
        data["diff_1"] =series.diff(periods=1)
        data['diff_1'].dropna()
        adf_test(data['diff_1']); 
        
    else:
        print("Please enter valid input")

# %%
non_stationarity_stationarity(df,df[target_col])

# %% [markdown]
# # ACF PACF
# 

# %%
# Plots for ACF and PACF #ask priti about ideal number to lags to be considered as default value
def acf_pacf(series,choice=15):



    #choice=input("Ideal Choice for lags are considered to be 10% to 30% of the length of the data.That means lags between 10 to 30 might be used, Please choose Accordingly.:\n")
    lags=int(choice)
    plt.rcParams.update({'figure.figsize': (20,6)})

    sm.graphics.tsa.plot_acf(series, lags=lags,title='auto correlation ',zero=False);
    sm.graphics.tsa.plot_pacf(series, lags=lags,title='partial auto correlation ',zero=False);
    print("Interpretation : \n ")
    print("""ACF represnts auto correlation between varibles w.r.t Time into consideration all components of time series.PACF represnts correlation function of the variables with residuals partially . \n""")
    print("Both ACF & PACF starts at lag 0 , which is the correlation of variables with itself and therefore results in a correlation of 1. Difference between both is inclusion and exclusion of indirect correlations. Blue area depicts 95% confidence interval.\n")
    print("CONCLUSION:\n")
    print( """ Sharp Drop Point: 
            Instant drop lag just after lag 0.

            ACF sharp drop point implies MA order & PACF sharp drop point implies AR order 

            Some basic approach for model choosing are as follows:

            1. ACF plot declines gradually and PACF drops instantly use AR model.
            2. ACF drops instantly and PACF declines gradually use MA model. 
            3. Both declines gradually use ARMA model
            4. Both drops instantly we are not able to model the time series.

            Note:

            ARIMA and SARIMA models are Intergrated ARMA models we will use the same identified orders from both the plots.


            """)

# %%
ACF_PACF(df[target_col])

# %% [markdown]
# # Train Test Split

# %%
# spliting dataset
def split(data):
    size_input=float(input("Please enter the size of percentage where you want to split the data-for eg 0.75 for 75% or 0.80 for 80%"))
    #splitting 85%/15% because of little amount of data
    size = int(len(data) * size_input)
    train= data[:size]
    test = data[size:]
    return(train,test)
    
    
    




# %%
train,test= split(df)

# %%
train.head()

# %%
train.shape

# %%
test.shape

# %%


# %%


# %% [markdown]
# # Forecasting 

# %% [markdown]
# ## AutoArima

# %%
# Autoarima model
def gen_auto_arima(df, col, m, f, periods, maxp=5, maxd=2, maxq=5, maxP=5, maxD=2, maxQ=5):
    automodel= auto_arima(df[col], seasonal=True, m=m, start_p=0, start_q=0, d=None, D=None, stepwise=True, max_p= maxp, max_d= maxd, max_q = maxq,
                         max_P= maxP, max_D= maxD, max_Q= maxQ)
    print(automodel.summary())
    preds, confint = automodel.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1], periods = periods, freq=f)
    fitted_series = pd.Series(preds, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    print(preds)
    plt.plot(df[target_col])
    plt.plot(fitted_series, color='darkgreen')
    fitted_series.to_excel('Output_forecast.xlsx')
    plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)
    plt.savefig('Forecast_autoARIMA.png')

# %%
gen_autoArima(train, target_col, 1, 'M', 98, maxp=5, maxd=2, maxq=5, maxP=5, maxD=2, maxQ=5)

# %%
def gen_auto_arima_plotly(df, col, m, f, periods, maxp=5, maxd=2, maxq=5, maxP=5, maxD=2, maxQ=5):
    automodel= auto_arima(df[col], seasonal=True, m=m, start_p=0, start_q=0, d=None, D=None, stepwise=True, max_p= maxp, max_d= maxd, max_q = maxq,
                         max_P= maxP, max_D= maxD, max_Q= maxQ)
    print(automodel.summary())
    preds, confint = automodel.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1], periods = periods, freq=f)
    fitted_series = pd.Series(preds, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    print(preds)
    fitted_series.to_excel('Output_forecast_plotly.xlsx')
    fitted_dataframe=pd.DataFrame(fitted_series,index=index_of_fc)
    pd.concat([train[target_col],fitted_dataframe],axis=1).iplot()

# %%
gen_autoArima_plotly(train, target_col, 1, 'M', 49, maxp=5, maxd=2, maxq=5, maxP=5, maxD=2, maxQ=5)

# %%
def naive_method(test_df):
    y_hat_naive = test_df.copy()
    y_hat_naive['naive_forecast'] = train[target_col][train_len-1]
    plt.figure(figsize=(12,4))
    plt.plot(train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
    plt.legend(loc='best')
    plt.title('Naive Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_naive['naive_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_naive['naive_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

# %%
def average_method(test_df):
    y_hat_average = test_df.copy()
    y_hat_average['average_forecast'] = train[target_col].mean()
    plt.figure(figsize=(12,4))
    plt.plot(train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_average['average_forecast'], label='Average forecast')
    plt.legend(loc='best')
    plt.title('Average Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_average['average_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_average['average_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Average method'], 'MAPE': [mape], 'RMSE': [rmse]})
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

# %%
def simple_moving_average(df, ma_window):
    y_hat_sma = df.copy()
    y_hat_sma['sma_forecast'] = data[target_col].rolling(ma_window).mean()
    y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]
    plt.figure(figsize=(12,4))
    plt.plot(train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
    plt.legend(loc='best')
    plt.title('Simple Moving Average Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_sma['sma_forecast'][train_len:])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_sma['sma_forecast'][train_len:])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })
    results = results[['Method', 'RMSE', 'MAPE']]
    return results

# %%
def simple_exponential_smoothing(test_df,forecast_duration):
    model = SimpleExpSmoothing(train[target_col])
    model_fit = model.fit(smoothing_level=0.2,optimized=False)
    model_fit.params
    y_hat_ses = test_df.copy()
    y_hat_ses['ses_forecast'] = model_fit.forecast(forecast_duration)
    plt.figure(figsize=(12,4))
    plt.plot(train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')
    plt.legend(loc='best')
    plt.title('Simple Exponential Smoothing Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_ses['ses_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_ses['ses_forecast'])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })
    return results

# %%
def holt_exponential_smoothing(test_df,seasonal_periods,forecast_duration):
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='additive', seasonal=None)
    model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)
    print(model_fit.params)
    y_hat_holt = test_df.copy()
    y_hat_holt['holt_forecast'] = model_fit.forecast(forecast_duration)
    plt.figure(figsize=(12,4))
    plt.plot( train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')
    plt.legend(loc='best')
    plt.title('Holt\'s Exponential Smoothing Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_holt['holt_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_holt['holt_forecast'])/test[target_col])*100,2)

    results = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })
    return results

# %%
def holtwinter_exponential_smoothing_additive(test_df,seasonal_periods,forecast_duration):    
    y_hat_hwa = test.copy()
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='add', seasonal='add')
    model_fit = model.fit(optimized=True)
    y_hat_hwa['hw_forecast'] = model_fit.forecast(forecast_duration)
    plt.figure(figsize=(12,4))
    plt.plot( train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')
    plt.legend(loc='best')
    plt.title('Holt Winters\' Additive Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_hwa['hw_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_hwa['hw_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })
    return results

# %%
def holtwinter_exponential_smoothing_multiplicative(test_df,seasonal_periods,forecast_duration):    
    y_hat_hwa = test.copy()
    model = ExponentialSmoothing(np.asarray(train[target_col]) ,seasonal_periods=seasonal_periods ,trend='add', seasonal='mul')
    model_fit = model.fit(optimized=True)
    y_hat_hwa['hw_forecast'] = model_fit.forecast(forecast_duration)
    plt.figure(figsize=(12,4))
    plt.plot( train[target_col], label='Train')
    plt.plot(test[target_col], label='Test')
    plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s multiplicative forecast')
    plt.legend(loc='best')
    plt.title('Holt Winters\' multiplicative Method')
    plt.show()
    rmse = np.sqrt(mean_squared_error(test[target_col], y_hat_hwa['hw_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(test[target_col]-y_hat_hwa['hw_forecast'])/test[target_col])*100,2)
    results = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })
    return results

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


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
            return df[col_name].fillna(df[col_name].ffill(),inplace=True)
        elif impute_method == "nocb" and (flag1 == False and flag2 == False and flag3 == False):
            return df[col_name].fillna(df[col_name].bfill(),inplace=True)

# %%


# %%


# %%


# %%



