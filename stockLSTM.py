import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("Stock Price Prediction using LSTM Model")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")  # Fetch last 5 years of data
    return df

if st.button("Predict"):
    # Fetch stock data
    df = fetch_stock_data(ticker)
    
    if df.empty:
        st.error("Invalid ticker or no data available. Please try again.")
    else:
        df = df[['Close']]
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)

        # Prepare training data
        X_train, y_train = [], []
        look_back = 60  # Use past 60 days to predict the future
        for i in range(look_back, len(scaled_data)):
            X_train.append(scaled_data[i-look_back:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict next 30 days
        last_look_back = scaled_data[-look_back:]  # Last 60 days
        last_look_back = np.reshape(last_look_back, (1, look_back, 1))

        future_forecast = []
        for _ in range(30):
            predicted_price = model.predict(last_look_back)[0, 0]
            future_forecast.append(predicted_price)
            new_input = np.append(last_look_back[:, 1:, :], [[[predicted_price]]], axis=1)
            last_look_back = new_input  # Update input for next prediction
        
        # Transform back to original scale
        future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

        # Generate future dates
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30, freq='D')

        # Plot Only Future Predictions
        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, future_forecast, label="Predicted Prices", linestyle='dashed', color="red")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Stock Price Prediction for {ticker} (Next 30 Days using LSTM)")
        plt.legend()

        # Display plot in Streamlit
        st.pyplot(plt)
