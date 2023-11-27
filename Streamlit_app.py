import streamlit as st
import MetaTrader5 as mt5
from Predictions import get_predictions,get_data
import pandas as pd
import numpy as np
import pandas_profiling
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
    with st.form("user_input"):
        currency_pairs = ["EURUSD","GBPUSD","USDCHF","USDJPY","USDCNH","USDRUB","AUDUSD","NZDUSD","USDCAD","USDSEK"]
        currency_pair = st.selectbox("Which currency pair would you like to forecast ?",currency_pairs)
        time_frames = ["1 Min","5 Min","15 Min","30 Min","1 Hour","4 Hour","1 Day","1 Week","1 Month"]
        time_frame = st.selectbox("Select the timeframe for forecasting",time_frames,index=4)
        method = st.radio("Select Forecasting method:",["ARIMA","Prophet","LSTM"])
        fine_tuning_methods = ["None","PmdArima","GridSearchCV"]
        fine_tuning = st.radio("Which fine tuning method you would like to use for ARIMA?",fine_tuning_methods,horizontal=True,index=0)
        submitted = st.form_submit_button("Forecast!")


if submitted:

    _,profile,_ = st.columns([1,3,1])

    with profile:

        st.title("Forex data (1000 records)")
        data = get_data(currency_pair,time_frame)
        st.dataframe(data)

    df = get_predictions(currency_pair,time_frame)

    _,chart,_ = st.columns([1,3,2])

    with chart:
        
        st.title("Forecasting Results")
        st.subheader("Actual Price")
        # Create a Seaborn pairplot
        plot = sns.lineplot(x=df.index,y=df["Actual"],)
        # Display the plot in Streamlit
        st.pyplot(plot.get_figure())
        plt.show()

        st.subheader("Forecasting results")
        st.line_chart(df)



    st.title("Exploratory Data Analysis")
    with st.spinner("Generating Profile Report"):
        profile_df = data.profile_report()
        st_profile_report(profile_df)


