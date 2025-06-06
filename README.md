# FinancialTradingBot

This project contains a full pipeline for building a stock trading agent, including data preprocessing, financial forecasting using machine learning models, trading simulation, and final agent evaluation.

---

# 🗂 Project Structure

- **`Data_Preprocess.ipynb`**: Preprocesses raw stock market data (e.g., calculates technical indicators, cleans data, prepares training features).
- **`Forecasting_Models.ipynb`**: Trains forecasting models (such as XGBoost, LSTM, or Prophet) to predict stock market trends or next-day closing prices.
- **`Simulation.ipynb`**: Simulates stock trading based on forecasted prices, testing different trading strategies and evaluating portfolio performance.
- **`FinalAgent.ipynb`**: Runs the final trading agent, integrating forecasting, risk management, and decision-making policies to simulate real-world trading scenarios.

---

# 🚀 How to Run

Clone this repository:

```bash
git clone https://github.com/YuLeWong2002/FinancialTradingBot.git
cd FinancialTradingBot
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks sequentially:

- Start with Data_Preprocess.ipynb to generate clean input data.
- Train forecasting models with Forecasting_Models.ipynb.
- Simulate trading strategies with Simulation.ipynb.
- Run full agent evaluation in FinalAgent.ipynb.

# 📜 License

This project is for educational and research purposes only.
