import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ===== Load all CSV logs from "logs/" folder =====
@st.cache_data
def load_all_logs(logs_dir="logs"):
    all_dfs = []
    for file_path in glob.glob(os.path.join(logs_dir, "ppo_trading_log_*.csv")):
        df = pd.read_csv(file_path, parse_dates=["Date"])
        # Extract stock name from filename
        stock_name = os.path.basename(file_path).replace("ppo_trading_log_", "").replace(".csv", "")
        df["Stock"] = stock_name
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

df = load_all_logs()

# ===== Sidebar Filters =====
st.sidebar.header("📊 Filter Options")

# 1. Select stock
stock_options = df["Stock"].unique()
selected_stock = st.sidebar.selectbox("Choose a stock", stock_options)

# 2. Select date range
min_date, max_date = df["Date"].min(), df["Date"].max()
start_date = st.sidebar.date_input("Start date", min_date)
end_date = st.sidebar.date_input("End date", max_date)

# ===== Filter data =====
filtered = df[
    (df["Stock"] == selected_stock) &
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# ===== Display result table =====
st.title("📈 PPO Trading Decisions Viewer")
st.write(f"**Stock:** {selected_stock}")
st.write(f"**Date Range:** {start_date} to {end_date}")
st.dataframe(filtered[["Date", "Action", "Price", "Balance", "Equity"]])

# ===== Custom Trading Strategy Plot =====
if not filtered.empty:
    st.subheader("📉 Trading Strategy Plot")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual price
    ax.plot(filtered["Date"], filtered["Price"], label="Actual Price", color="green", linewidth=1, linestyle='--')

    # Plot predicted next close line
    ax.plot(filtered["Date"], filtered["Forecasted_Price"], label="Forecasted_Price", color="black", linewidth=1)

    # Add Buy/Sell markers
    buy_signals = filtered[filtered["Action"] == "Buy"]
    sell_signals = filtered[filtered["Action"] == "Sell"]

    ax.scatter(buy_signals["Date"], buy_signals["Forecasted_Price"], color="blue", label="Buy", marker='o')
    ax.scatter(sell_signals["Date"], sell_signals["Forecasted_Price"], color="red", label="Sell", marker='o')

    ax.set_title(f"Predicted Price Trading Strategy for {selected_stock}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Next Close Price")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.warning("⚠️ No data available in the selected date range.")

