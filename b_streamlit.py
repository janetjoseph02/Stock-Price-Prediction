import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime
import numpy as np
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import requests


st.title("Indian Stock Prediction App")

# Function to load stock data
@st.cache_data
def load_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

#Function to fetch news data
@st.cache_data
def get_news_data(company_name):
    # Set up your NewsAPI key and base URL
    newsapi_key = '1d048dec9c29473babb88c80edd148ca'
    base_url = 'https://newsapi.org/v2/everything'

    params = {
        'q': company_name,
        'apiKey': newsapi_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 10  # Limit to 20 articles
    }

    response = requests.get(base_url, params=params)
    news_data = response.json()
    return news_data

# Function to display dynamic line chart using Plotly
def plot_stock_data(stock_data):
    st.markdown("### Stock Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig.update_layout( 
        title="",
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        template="plotly_dark"
    )
    st.plotly_chart(fig)

# Function to display price movements
def display_prices(stock_data):
    st.header("Price Movements")
    stock_data["% Change"] = (stock_data["Adj Close"] / stock_data["Adj Close"].shift(1) - 1) * 100 
    stock_data.dropna(inplace = True)
    # stock_data["% Change"] = stock_data["% Change"].apply(lambda x: f"{x:.2f}%")
    st.write(stock_data)
    annual_return = stock_data["% Change"].mean() * 252 * 100
    st.write("Annual return is ", annual_return, "%")
    stdev = np.std(stock_data["% Change"]) * np.sqrt(252)
    st.write("Standard Deviation is ", stdev*100, "%")
    st.write("Risk-Adjusted Return is ", annual_return/(stdev*100))

# Function to display news
def display_news(news_data):    
    # Check if the response is successful
    if news_data['status'] == 'ok':
        # Extract the news articles from the response
        articles = news_data['articles']

        # Loop through the articles to display them
        for i in range(len(articles)):
            with st.expander(f"News {i+1}"):
                # Display the article details
                st.write("Published on : ", articles[i]['publishedAt'])
                st.write("Title : ", articles[i]['title'])
                st.write("Summary : ", articles[i]['description'])
                st.write(f"URL: {articles[i]['url']}")
                # Display the image if available
                if 'urlToImage' in articles[i] and articles[i]['urlToImage']:
                    st.image(articles[i]['urlToImage'], caption="Article Image", use_column_width=True)

# Function to create and train XGBoost model
def create_xgboost_model(stock_data, n_steps=60):
    # Use the 'Close' prices for XGBoost
    prices = stock_data['Close'].values

    # Prepare the training data
    X, y = [], []
    for i in range(n_steps, len(prices)):
        X.append(prices[i-n_steps:i])
        y.append(prices[i])
    X, y = np.array(X), np.array(y)

    # Train XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    return model

# Function to evaluate XGBoost and predict future prices
def evaluate_and_predict_xgboost(model, actual_prices, n_steps=60, future_days=30):
    X_test = []
    scaled_prices = actual_prices.values

    for i in range(n_steps, len(scaled_prices)):
        X_test.append(scaled_prices[i-n_steps:i])

    X_test = np.array(X_test)

    # Predict the prices
    predicted_prices = model.predict(X_test)

    # Create DataFrame for predicted prices
    predicted_prices_df = pd.DataFrame(predicted_prices, index=actual_prices.index[n_steps:], columns=['XGBoost Predictions'])

    # Future Price Prediction
    future_predictions = []
    last_n_days = scaled_prices[-n_steps:]

    for _ in range(future_days):
        future_predictions.append(model.predict(last_n_days.reshape(1, -1))[0])
        last_n_days = np.append(last_n_days[1:], future_predictions[-1])

    future_dates = pd.date_range(start=actual_prices.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_prices_df = pd.DataFrame(future_predictions, index=future_dates, columns=['XGBoost Future Predictions'])

    return predicted_prices_df, future_prices_df


# Function to calculate accuracy metrics
def calculate_accuracy(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mape, r2

# Function to plot the best model's performance
def plot_model(model_name, ticker_name, actual_prices, predicted_prices_df, future_prices_df, mse, rmse, mape, r2):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color='blue')
    plt.plot(predicted_prices_df, label=f"{model_name} Predicted Prices", color='red')
    plt.plot(future_prices_df, label=f"{model_name} Future Prices", color='green')
    plt.title(f"{model_name} Model for {ticker_name}\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    st.pyplot(plt)

# Streamlit UI setup

# Sidebar inputs
st.sidebar.header("Input Parameters")
exchange = st.sidebar.selectbox("Select Stock Exchange", ["NSE", "BSE"])  # Dropdown to select the stock exchange
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE, TCS, INFY):").upper()

start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Select End Date", pd.to_datetime(datetime.date.today()), min_value=start_date ,max_value=pd.to_datetime(datetime.date.today()))


# Determine the correct stock ticker suffix based on the selected exchange
if exchange == "NSE":
    ticker += ".NS"
elif exchange == "BSE":
    ticker += ".BO"

ticker_symbol = yf.Ticker(ticker)
# print('ticker -> ',ticker)
company_name = ticker_symbol.info.get('shortName', 'No Name Available')
# company_name = ticker_symbol.info['shortName']


# Load and display stock data
if st.sidebar.button("Fetch Stock Data"):
    stock_data = load_data(ticker, start_date, end_date)
    news_data = get_news_data(company_name)
    st.write(f"### {company_name}")

    # Plot stock data dynamically
    plot_stock_data(stock_data)

    actual_prices = stock_data['Close']
    # XGBoost predictions
    xgboost_model = create_xgboost_model(stock_data)
    xgboost_predictions_df, xgboost_future_df = evaluate_and_predict_xgboost(xgboost_model, actual_prices, future_days=30)

    # Calculate accuracy for XGBoost
    xgboost_mse, xgboost_rmse, xgboost_mape, xgboost_r2 = calculate_accuracy(actual_prices[xgboost_predictions_df.index], xgboost_predictions_df['XGBoost Predictions'])

    # plot model 
    plot_model("XGBoost", ticker, actual_prices, xgboost_predictions_df, xgboost_future_df, xgboost_mse, xgboost_rmse, xgboost_mape, xgboost_r2)

    tabs = st.tabs(["Pricing Data", "News"])

    with tabs[0]:
        display_prices(stock_data)

    with tabs[1]:
        st.write("### Stock News")
        display_news(news_data)  # Call the function to display news