import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from datetime import datetime

def create_lstm_model(ticker_symbol, start_date='2020-01-01', end_date=None, n_steps=60, epochs=50, batch_size=32):
    """
    Fetch stock data for the given ticker symbol and create and train an LSTM model.
    
    Parameters:
    - ticker_symbol: str, ticker symbol of the stock (e.g., 'RELIANCE.NS').
    - start_date: str, start date for historical data (format: 'YYYY-MM-DD').
    - end_date: str, end date for historical data (default: today).
    - n_steps: int, number of time steps to look back for predictions.
    - epochs: int, number of training epochs.
    - batch_size: int, size of the training batches.
    
    Returns:
    - model: trained Keras LSTM model.
    - scaler: MinMaxScaler fitted on training data for future inverse transformation.
    - actual_prices: DataFrame of actual closing prices.
    """
    
    # Set end_date to today if not provided
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Fetch the historical stock data
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    # Use the 'Close' prices for LSTM
    prices = data['Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)

    # Prepare the training data
    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_data)):
        X_train.append(scaled_data[i-n_steps:i, 0])  # Previous n_steps
        y_train.append(scaled_data[i, 0])           # Target value
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape for LSTM (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model, scaler, data['Close']

def evaluate_and_predict(model, scaler, actual_prices, ticker_symbol, n_steps=60, future_days=30):
    """
    Evaluate the LSTM model on the training data and predict future stock prices.
    
    Parameters:
    - model: Trained LSTM model.
    - scaler: Fitted MinMaxScaler for inverse transformation.
    - actual_prices: Actual closing prices used in the model.
    - ticker_symbol: Stock symbol for display.
    - n_steps: Number of time steps used in the LSTM model.
    - future_days: Number of days to predict into the future.
    """

    # Predict the prices for the training period
    scaled_prices = scaler.transform(actual_prices.values.reshape(-1, 1))
    
    X_test = []
    for i in range(n_steps, len(scaled_prices)):
        X_test.append(scaled_prices[i-n_steps:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict the prices for the training period
    predicted_prices_scaled = model.predict(X_test)
    
    # Inverse transform to get predicted prices in the original scale
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    # Create a DataFrame for the predicted prices (aligned with the actual prices)
    predicted_prices_df = pd.DataFrame(predicted_prices, index=actual_prices.index[n_steps:], columns=['Predicted Price'])

    # Combine actual and predicted prices for plotting
    combined_df = pd.concat([actual_prices[n_steps:], predicted_prices_df], axis=1)

    # --- Future Price Prediction ---
    # Prepare the last n_steps of actual prices for future predictions
    last_n_days_scaled = scaled_prices[-n_steps:]
    
    future_predictions = []

    # Predict future prices one step at a time
    for _ in range(future_days):
        X_future = []
        X_future.append(last_n_days_scaled)
        X_future = np.array(X_future)
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

        # Predict the next day's price
        predicted_price_scaled = model.predict(X_future)
        
        # Inverse transform the predicted price
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        # Append the prediction and update the last_n_days_scaled
        future_predictions.append(predicted_price[0][0])
        last_n_days_scaled = np.append(last_n_days_scaled[1:], predicted_price_scaled, axis=0)

    # Create future dates for the predictions
    future_dates = pd.date_range(start=actual_prices.index[-1] + pd.Timedelta(days=1), periods=future_days)
    
    # Create a DataFrame for the future predicted prices
    future_prices_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predicted Price'])

    # Combine actual, training predictions, and future predictions
    combined_df_with_future = pd.concat([combined_df, future_prices_df], axis=0)

    # Plot actual vs predicted prices (training and future)
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices.index, actual_prices, label='Actual Prices', color='blue', linewidth=1)
    plt.plot(combined_df.index, combined_df['Predicted Price'], label='Predicted Prices (Training)', color='red', linewidth=1)
    plt.plot(future_prices_df.index, future_prices_df['Future Predicted Price'], label='Future Predicted Prices', color='green', linewidth=1)
    plt.title(f"Stock Price Prediction for {ticker_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

ticker = 'RELIANCE.NS'
model, scaler, actual_prices = create_lstm_model(ticker, start_date='2020-01-01', epochs=100, batch_size=32)
evaluate_and_predict(model, scaler, actual_prices, ticker, future_days=30)
