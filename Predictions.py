from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
import datetime
import streamlit as st

# Initialize MT5 connection
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

def get_predictions(currency_pair,timeframe):

    time_freq_dict = {
    "1 Min": "T",
    "5 Min": "5T",
    "15 Min": "15T",
    "30 Min": "30T",
    "1 Hour": "H",
    "4 Hour": "4H",
    "1 Day": "D",
    "1 Week": "W",
    "1 Month": "M"}

    timeframes = {"1 Min":mt5.TIMEFRAME_M1,"5 Min":mt5.TIMEFRAME_M5,
                  "15 Min":mt5.TIMEFRAME_M15,"30 Min":mt5.TIMEFRAME_M30,
                  "1 Hour":mt5.TIMEFRAME_H1,"4 Hour":mt5.TIMEFRAME_H4,
                  "6 Hour":mt5.TIMEFRAME_H6,"1 Day":mt5.TIMEFRAME_D1,
                  "1 Week":mt5.TIMEFRAME_W1,"1 Month":mt5.TIMEFRAME_MN1}

    # Get historical data
    rates = mt5.copy_rates_from_pos(currency_pair, timeframes[timeframe], 0, 1000)

    df = pd.DataFrame(rates)

    df.columns = ["Timestamp","Open","High","Low","Close","Tick Volume","Spread","Real Volume"]
    df.index = df.Timestamp.apply(lambda a : datetime.datetime.fromtimestamp(a))
     
    # Use 'close' column for ARIMA model
    model = ARIMA(df["Close"], order=(1,3,1))

    # Fit the model
    model_fit = model.fit()

    past = model_fit.predict()


    # Forecast
    forecast = model_fit.forecast(steps=30)
    idx = pd.date_range(df.index[-1], periods=31, freq=time_freq_dict[timeframe])[1:]
    forecast.index = idx


    predicted = pd.concat([past,forecast])
    predicted[0] = np.nan


    actual = df.Close.values
    difference = len(predicted)-len(actual)
    fill = [np.nan for i in range(difference)]
    actual = np.concatenate([actual,fill])

    df_forecast = pd.DataFrame({"Predicted":predicted.values,"Actual":actual},index=predicted.index)

    return df_forecast

def get_data(currency_pair,timeframe):
    
    timeframes = {"1 Min":mt5.TIMEFRAME_M1,"5 Min":mt5.TIMEFRAME_M5,
                  "15 Min":mt5.TIMEFRAME_M15,"30 Min":mt5.TIMEFRAME_M30,
                  "1 Hour":mt5.TIMEFRAME_H1,"4 Hour":mt5.TIMEFRAME_H4,
                  "6 Hour":mt5.TIMEFRAME_H6,"1 Day":mt5.TIMEFRAME_D1,
                  "1 Week":mt5.TIMEFRAME_W1,"1 Month":mt5.TIMEFRAME_MN1}

    # Get historical data
    rates = mt5.copy_rates_from_pos(currency_pair, timeframes[timeframe], 0, 1000)

    df = pd.DataFrame(rates)

    df.columns = ["Timestamp","Open","High","Low","Close","Tick Volume","Spread","Real Volume"]

    df.Timestamp = df.Timestamp.apply(lambda a : datetime.datetime.fromtimestamp(a))

    return df

