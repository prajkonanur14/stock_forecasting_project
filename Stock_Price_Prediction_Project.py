####################### Stock Price Prediction Project ########################
###Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
#Import Plotly Libraries
import plotly.express as px
import plotly.graph_objects as go
#from dash import Dash, dcc, html, Input, Output
import warnings
warnings.simplefilter(action='ignore') 

###Import Data
df_mck = pd.read_csv('MCK.csv')   #McKesson Corporation
df_vz = pd.read_csv('VZ.csv')     #Verizon Communications Inc.
df_xom = pd.read_csv('XOM.csv')   #Exxon Mobil Corporation
df_hlt = pd.read_csv('HLT.csv')   #Hilton Worldwide Holdings Inc.
df_meta = pd.read_csv('META.csv') #META Platforms Inc.

###Data Pre-Processing
#Convert Date from Object to DateTime
stock_list = [df_mck, df_vz, df_xom, df_hlt, df_meta]
for item in stock_list:
    item['Date'] = pd.to_datetime(item['Date'])
#Splitting Data into Train and Test Dataframes
df_split = len(df_mck)
#Train Data
df_mck_train = df_mck[:int(df_split * 0.9)]
df_vz_train = df_vz[:int(df_split * 0.9)]
df_xom_train = df_xom[:int(df_split * 0.9)]
df_hlt_train = df_hlt[:int(df_split * 0.9)]
df_meta_train = df_meta[:int(df_split * 0.9)]
#Test Data
df_mck_test = df_mck[int(df_split * 0.9):]
df_vz_test = df_vz[int(df_split * 0.9):]
df_xom_test = df_xom[int(df_split * 0.9):]
df_hlt_test = df_hlt[int(df_split * 0.9):]
df_meta_test = df_meta[int(df_split * 0.9):]


#Display Closing Prices
def closing_price():
    #Set Date as Index
    for item in stock_list:
        item.set_index('Date', inplace=True)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df_mck['Close'],
            color='blue')
    ax.plot(df_vz['Close'],
            color='firebrick')
    ax.plot(df_xom['Close'],
            color='grey')
    ax.plot(df_hlt['Close'],
            color='orange')
    ax.plot(df_meta['Close'],
            color='limegreen')
    #Set Labels
    ax.set_title('Daily Closing Prices for MCK, VZ, XOM, HLT, META')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Closing Price')
    ax.legend(['MCK', 'VZ', 'XOM', 'HLT', 'META'], loc='upper left')
    ax.grid()

#Augmented Dickey-Fuller Test Function
def adf_test(data):
    adf,pvalue,usedlag,nobs,critical_values,icbest = adfuller(data)
    print(f"1. adf value = {adf}.")
    print(f"2. P-Value = {pvalue}.")
    print(f"3. Number of lags = {usedlag}")
    print(f"4. Num Of Observations Used For ADF Regression and Critical Values Calculation = {nobs}")
    print("5. Critical Values: ")
    for key, val in critical_values.items():
        print("\t",key,": ", val)
#Partial Autocorrelation Plot Function (p)
def pacf(data):
    plot_pacf(data, lags=20)
#Autocorrelation Plot Function (q)
def acf(data):
    plot_acf(data, lags=20)
#Seasonal Decompose Function
def decompose_plot(data):
    
    decompose = seasonal_decompose(data, model='multiplicative', period=30)
    #decompose.plot()
    est_trend = decompose.trend
    est_seasonal = decompose.seasonal
    est_residual = decompose.resid
    return est_residual

###Technical Indicators (Use df_stock dataframes)
#Exponential Moving Average (EMA)
def ema(data, period=20):
    return data['Close'].ewm(span=period, adjust=False).mean()
#Moving Average Convergence Difference (MACD)
def macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema-slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    histogram = macd_line-signal_line
    return macd_line
#On-Balance Volume (OBV)
def obv(data):
    close = data['Close']
    volume = data['Volume']
    obv = np.where(close > close.shift(), volume, np.where(close < close.shift(), 
                                                           -volume, 0)).cumsum()
    return obv


###ARIMA Model Functions
#Prophet Model Function (Use df_stock dataframes)
def prophet_forecast(data, periods=365):
    #Pre-Processing
    data.reset_index(inplace=True)
    data_proph = data[['Date', 'Close']]
    data_proph.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)
    #Forecasting Using Prophet
    prophet = Prophet(yearly_seasonality='auto')
    prophet.fit(data_proph)
    #Predict Prophet Model
    future_close = prophet.make_future_dataframe(periods=periods)
    future_forecast = prophet.predict(future_close)
    future_key_stats = future_forecast[['ds','yhat','yhat_lower','yhat_upper']]
    #Plot Prophet Model
    fig_1 = prophet.plot(future_forecast, 
                         figsize=(12,8), 
                         xlabel='Date', 
                         ylabel='Daily Closing Price', 
                         include_legend=True)
    a = add_changepoints_to_plot(fig_1.gca(), prophet, future_forecast)
    #Plot Components of Prophet Model
    fig_2 = prophet.plot_components(future_forecast)
    return future_key_stats

#AutoArima Model Function (Use stock_close dataframes)
def auto_a(data):
    model = auto_arima(data['Close'],
                       test='adf',       #Use adf test
                       seasonal=False,   #No seasonality in model
                       trace= True, 
                       suppress_warnings=True)
    print(model.summary())
    model.plot_diagnostics(figsize=(15,8))

#ARIMA Model Function 
def arima_model(train, test, order=(0,1,0)):
    #Define Arima Model
    def arima_forecast(history):
        #Fit Model
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        
        #Make Prediction
        output = model_fit.forecast()
        yhat = output[0]
        return yhat
    
    train = train.values
    test = test.values
    #Walk Forward Validation
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        #Generate a Prediction
        yhat2 = arima_forecast(history)
        predictions.append(yhat2)
        #Add Predicted Value to Training Set
        obs = test[t]
        history.append(obs)
    return predictions


###Evaluation Metrics Calculation
def eval_metrics(data):
    resid = data['Close'] - data['Predicted Close']
    mae = mean_absolute_error(data['Close'], data['Predicted Close'])
    mse = mean_squared_error(data['Close'], data['Predicted Close'])
    return (resid, mae, mse)

#Plotting Test Data vs Predicted Data
def final_plot(data, stock_name, color1, color2='black'):
    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    ax[0].plot(data['Close'], color=color1)
    ax[1].plot(data['Predicted Close'], color=color2)
    #Labels
    ax[0].set_title(f'Closing Stock price for {stock_name}')
    ax[1].set_title(f'Predicted Closing Stock Price for {stock_name}')
    ax[0].set_ylabel('Closing Stock Price')
    fig.tight_layout()
    ax[0].grid()
    ax[1].grid()

###Stationarize Data
train_list = [df_mck_train, df_vz_train, df_xom_train, df_hlt_train, df_meta_train]
test_list = [df_mck_test, df_vz_test, df_xom_test, df_hlt_test, df_meta_test]
#Set Date as Index for Train Data
for item in train_list:
    item.set_index('Date', inplace=True)
#Set Date as Index for Test Data
for item in test_list:
    item.set_index('Date', inplace=True)

#Train Series for Stock Close Price
mck_close,vz_close,xom_close,hlt_close,meta_close = (df_mck_train['Close'].to_frame(), 
                                                     df_vz_train['Close'].to_frame(), 
                                                     df_xom_train['Close'].to_frame(), 
                                                     df_hlt_train['Close'].to_frame(), 
                                                     df_meta_train['Close'].to_frame())
#Test Series for Stock Close Price
mck_test,vz_test,xom_test,hlt_test,meta_test = (df_mck_test['Close'].to_frame(),  
                                                df_vz_test['Close'].to_frame(), 
                                                df_xom_test['Close'].to_frame(), 
                                                df_hlt_test['Close'].to_frame(), 
                                                df_meta_test['Close'].to_frame())

#Stationary Transformations for Train Data
#Log Transformation
mck_close_log = np.log(mck_close)
vz_close_log = np.log(vz_close)
xom_close_log = np.log(xom_close)
hlt_close_log = np.log(hlt_close)
meta_close_log = np.log(meta_close)
#Differencing by Order of 1
mck_close_log_diff = mck_close_log.diff().dropna()
vz_close_log_diff = vz_close_log.diff().dropna()
xom_close_log_diff = xom_close_log.diff().dropna()
hlt_close_log_diff = hlt_close_log.diff().dropna()
meta_close_log_diff = meta_close_log.diff().dropna()
#Stationary Transformations for Test Data
#Log Transformation
mck_test_log = np.log(mck_test)
vz_test_log = np.log(vz_test)
xom_test_log = np.log(xom_test)
hlt_test_log = np.log(hlt_test)
meta_test_log = np.log(meta_test)
#Differencing by Order of 1
mck_test_log_diff = mck_test_log.diff().dropna()
vz_test_log_diff = vz_test_log.diff().dropna()
xom_test_log_diff = xom_test_log.diff().dropna()
hlt_test_log_diff = hlt_test_log.diff().dropna()
meta_test_log_diff = meta_test_log.diff().dropna()



###Calling Models for Each Stock and Gathering Evaluation Metric Data
##McKesson##
#mck_solution = arima_model(mck_close, mck_test, order=(2,1,2))
#mck_test.insert(1, "Predicted Close", mck_solution, True)
#resid_mck,mae_mck,mse_mck = eval_metrics(mck_test)

#mck_solution2 = arima_model(mck_close_log_diff, mck_test_log_diff, order=(3,0,4))
#mck_test_log_diff.insert(1, "Predicted Close", mck_solution2, True)
#resid_2_mck,mae_2_mck,mse_2_mck = eval_metrics(mck_test_log_diff)


##Verizon##
#vz_solution = arima_model(vz_close, vz_test, order=(1,1,4))
#vz_test.insert(1, "Predicted Close", vz_solution, True)
#resid_vz,mae_vz,mse_vz = eval_metrics(vz_test)

#vz_solution2 = arima_model(vz_close_log_diff, vz_test_log_diff, order=(2,0,0))
#vz_test_log_diff.insert(1, "Predicted Close", vz_solution2, True)
#resid_2_vz,mae_2_vz,mse_2_vz = eval_metrics(vz_test_log_diff)


##Exxon Mobil##
#xom_solution = arima_model(xom_close, xom_test, order=(0,1,0))
#xom_test.insert(1, "Predicted Close", xom_solution, True)
#resid_xom,mae_xom,mse_xom = eval_metrics(xom_test)

#xom_solution2 = arima_model(xom_close_log_diff, xom_test_log_diff, order=(2,0,2))
#xom_test_log_diff.insert(1, "Predicted Close", xom_solution2, True)
#resid_2_xom,mae_2_xom,mse_2_xom = eval_metrics(xom_test_log_diff)


##Hilton##
#hlt_solution = arima_model(hlt_close, hlt_test, order=(2,1,2))
#hlt_test.insert(1, "Predicted Close", hlt_solution, True)
#resid_hlt,mae_hlt,mse_hlt = eval_metrics(hlt_test)

#hlt_solution2 = arima_model(hlt_close_log_diff, hlt_test_log_diff, order=(1,0,0))
#hlt_test_log_diff.insert(1, "Predicted Close", hlt_solution2, True)
#resid_2_hlt,mae_2_hlt,mse_2_hlt = eval_metrics(hlt_test_log_diff)


#META##
#meta_solution = arima_model(meta_close, meta_test, order=(3,1,1))
#meta_test.insert(1, "Predicted Close", meta_solution, True)
#resid_meta,mae_meta,mse_meta = eval_metrics(meta_test)

#meta_solution2 = arima_model(meta_close_log_diff, meta_test_log_diff, order=(0,0,0))
#meta_test_log_diff.insert(1, "Predicted Close", meta_solution2, True)
#resid_2_meta,mae_2_meta,mse_2_meta = eval_metrics(meta_test_log_diff)









