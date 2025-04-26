# FinancialTradingBot

This project contains a full pipeline for building a stock trading agent, including data preprocessing, financial forecasting using machine learning models, and trading simulation.

# 🗂 Project Structure

Data_Preprocess.ipynb:	Preprocesses raw stock market data (e.g., calculates technical indicators, cleans data, prepares training features).

Forecasting_Models.ipynb:	Trains forecasting models (such as XGBoost, LSTM, or Prophet) to predict stock market trends or next-day closing prices.

Simulation.ipynb:	Simulates stock trading based on forecasted prices, testing different trading strategies and evaluating portfolio performance.

# 🚀 How to Run
Clone this repository:

git clone https://github.com/YuLeWong2002/FinancialTradingBot.git

cd FinancialTradingBot

Install required dependencies

Start with Data_Preprocess.ipynb to generate clean input data.

Train forecasting models with Forecasting_Models.ipynb.

Evaluate trading strategies in Simulation.ipynb.

# 📊 Features
Data cleaning and feature engineering (moving averages, RSI, MACD, etc.)

Forecasting stock prices with machine learning models

Simulating real-world trading environments

Portfolio performance tracking

# 📦 Requirements
You can create a virtual environment and install the necessary packages:

pip install pandas numpy matplotlib scikit-learn xgboost
You may also need:

prophet (for time series forecasting)

torch (for deep learning models like LSTM)

# 📜 License
This project is for educational and research purposes.

