from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import requests
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# from prophet import Prophet


def get_predictions_ARIMA(currency_pair,timeframe,fine_tuning_method,API_KEY):

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

    df = get_data(currency_pair,timeframe,API_KEY)

    if fine_tuning_method=="PmdArima":
        # Fit a simple auto_arima model
        model = pm.auto_arima(df["Close"].astype(float), seasonal=True, m=12)
        order = model.order
    else:
        order=(1,3,1)
     
    # Use 'close' column for ARIMA model
    model = ARIMA(df["Close"].astype(float), order=order)

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

# def get_predictions_prophet(currency_pair,timeframe,API_KEY):

#     df = get_data(currency_pair,timeframe,API_KEY)
#     df["Timeframe"]=df.index
#     df = df.rename(columns={'Timeframe': 'ds', 'Close': 'y'})
#     df['ds'] = df['ds'].dt.tz_localize(None) 

#     # Create a Prophet model
#     model = Prophet()
#     model.fit(df.astype(float))

#     # Specify the number of periods to forecast into the future
#     n_periods = 30  # Replace with the number of periods you want to forecast
    
#     past = model.predict()
#     # Create a DataFrame with future dates
#     future = model.make_future_dataframe(periods=n_periods)

#     # Use the fitted model to make forecasts
#     forecast = model.predict(future)
    
#     # Extract the forecasted values and relevant columns
#     forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

#     return forecast_df["yhat"]

# def get_predictions_LSTM(currency_pair,timeframe,API_KEY):

#     time_freq_dict = {
#     "1 Min": "T",
#     "5 Min": "5T",
#     "15 Min": "15T",
#     "30 Min": "30T",
#     "1 Hour": "H",
#     "4 Hour": "4H",
#     "1 Day": "D",
#     "1 Week": "W",
#     "1 Month": "M"}

#     df = get_data(currency_pair,timeframe,API_KEY)

#     # Initialize Min-Max Scaler
#     scaler = MinMaxScaler()

#     target = pd.DataFrame(data=df["Close"],columns=["Close"])

#     # Scale the target variable using the scaler
#     target["Scaled Data"] = scaler.fit_transform(target[["Close"]])

#     X=[]
#     y=[]
#     # Extract 100 records as independent values and 101st record as target value over each iteration
#     for i in range(len(target)-100):
#         X.append(list(target.iloc[i:i+100,1]))
#         y.append(target.iloc[i+100,1])

#     # Initialize a sequential model using LSTM algorithm
#     model=Sequential()
#     model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
#     model.add(LSTM(50,return_sequences=True))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error',optimizer='adam')

#     # Train model using train and test data
#     model.fit(X,y,validation_data=(X,y),epochs=100,batch_size=64,verbose=1)

#     # Store input values of prediction
#     predict_input = target.iloc[-100:,1].values.tolist()
#     # Create empty list to store output values of prediction
#     predict_output = []
#     # Create list to store both input and output data
#     predict_data = predict_input.copy()

#     # Iterate for loop to predict prices of 30 successive days
#     for i in range(30):
#         # Pass 100 values as input
#         step_input = predict_data[-100:]
#         # Predict 101st value using developed model
#         step_output = model.predict([list(step_input)],verbose=0).reshape(1)
#         # Add the predicted value to the input value
#         predict_data.extend(step_output)
#         # Store the output values in a list
#         predict_output.extend(step_output)

#     # Transform output data
#     predicted = scaler.inverse_transform(np.array(predict_output).reshape(-1,1)).reshape(-1)
#     forecasted = np.concatenate([df["Close"].values,predicted])
#     difference = len(forecasted)-len(df)
#     fill = [np.nan for i in range(difference)]
#     actual = np.concatenate([df["Close"].values,fill])

#     df_forecast = pd.DataFrame({"Predicted":forecasted,"Actual":actual})
#     new_idx = pd.date_range(df.index[-1], periods=31, freq=time_freq_dict[timeframe])[1:]
#     idx = np.concatenate([df.index,new_idx])
#     df_forecast.index = idx
#     st.dataframe(df_forecast)
#     return df_forecast

def get_data(currency_pair,timeframe,API_KEY):
    
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    timeframes = {"1 Min":"function=TIME_SERIES_INTRADAY&interval=1min",
                  "5 Min":"function=TIME_SERIES_INTRADAY&interval=5min",
                  "15 Min":"function=TIME_SERIES_INTRADAY&interval=15min",
                  "30 Min":"function=TIME_SERIES_INTRADAY&interval=30min",
                  "1 Hour":"function=TIME_SERIES_INTRADAY&interval=60min",
                  "1 Day":"function=TIME_SERIES_DAILY",
                  "1 Week":"function=TIME_SERIES_WEEKLY",}
    timeseries = {"1 Week":"Weekly Time Series","1 Day":"Time Series (Daily)",
                  "1 Hour":"Time Series (60min)","30 Min":"Time Series (30min)",
                  "15 Min":"Time Series (15min)","5 Min":"Time Series (5min)",
                  "1 Min":"Time Series (1min)"}
    url = "https://www.alphavantage.co/query?"+timeframes[timeframe]+"&symbol="+currency_pair+"&apikey="+API_KEY+"&outputsize=full"
    r = requests.get(url)
    data = r.json()

    df = pd.DataFrame(data[timeseries[timeframe]]).T[::-1][:1000]
    df.columns = ["Open","High","Low","Close","Volume"]

    for col in df.columns:
        df[col] = df[col].astype(float)

    return df

