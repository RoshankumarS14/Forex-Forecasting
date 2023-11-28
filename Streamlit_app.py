import streamlit as st
from Predictions import get_predictions_ARIMA,get_predictions_LSTM,get_data
import pandas as pd
import numpy as np
# import pandas_profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Forex Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_,form,_ = st.columns([1,3,1])

with form:
    st.title("Forex Forecasting")
    # Generate quiz
    st.write("""
    Ever looked a forex chart and wondered how well you understood its trend? Here's a fun twist: Instead of just looking at chart, come to **Forex Forecasting** and predict the prices!

    **How does it work?** 🤔
    1. Select the currency pair.
    2. Select the timeframe in which you want to trade.
    3. Select the model which you want to use to forecast the market.         
    4. Enter your [Alpha Advantage API Key].
       Claim your free alpha advantage API key from here.(https://www.alphavantage.co/support/#api-key).

    ⚠️ Disclaimer: These predictions are purely for educational purpose.

    Once you've input the details, voilà! Dive deep into predictions and analysis of the market! 
    """)
    with st.form("user_input"):
        currency_pairs = ["EURUSD","GBPUSD","USDCHF","USDJPY","USDCNH","USDRUB","AUDUSD","NZDUSD","USDCAD","USDSEK"]
        currency_pair = st.selectbox("Which currency pair would you like to forecast ?",currency_pairs)
        time_frames = ["1 Min","5 Min","15 Min","30 Min","1 Hour","1 Day","1 Week"]
        time_frame = st.selectbox("Select the timeframe for forecasting",time_frames,index=4)
        method = st.radio("Select Forecasting method:",["ARIMA","Prophet","LSTM"])
        fine_tuning_methods = ["None","PmdArima","GridSearchCV"]
        fine_tuning = st.radio("Which fine tuning method you would like to use for ARIMA?",fine_tuning_methods,horizontal=True,index=0)
        API_KEY = st.text_input("Enter your Alpha Advantage API Key:", placeholder="sk-XXXX", type='password')
        submitted = st.form_submit_button("Forecast!")


if submitted:

    _,profile,_ = st.columns([1,3,1])

    with profile:

        data = get_data(currency_pair,time_frame,API_KEY)
        st.title(currency_pair+" data ("+str(len(data))+" records)")
        st.dataframe(data)

    if method=="ARIMA":
        df = get_predictions_ARIMA(currency_pair,time_frame,fine_tuning,API_KEY)
    elif method=="LSTM":
        df = get_predictions_LSTM(currency_pair,time_frame,API_KEY)
    else: 
        df = get_predictions_ARIMA(currency_pair,time_frame,API_KEY)

    _,chart,_ = st.columns([1,3,2])

    with chart:
        
        st.title("Forecasting Results")
        st.subheader("Actual Price")

        df["Predicted"]=df["Predicted"].astype(float)
        df["Actual"]=df["Actual"].astype(float)

        # Create a Seaborn pairplot
        plot = sns.lineplot(x=np.arange(len(df)),y=df["Actual"].astype(float))
        # Display the plot in Streamlit
        st.pyplot(plot.get_figure())
        plt.show()

        st.subheader("Forecasting results")
        
        st.line_chart(df)



    st.title("Exploratory Data Analysis")
    with st.spinner("Generating Profile Report"):
        profile_df = data.profile_report()
        st_profile_report(profile_df)


