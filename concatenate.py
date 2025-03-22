import pandas as pd
from dateutil import parser

# Define input file names
input_files = [
    "Walt_Disney_Company_news_sentiment_1.csv",
    "Walt_Disney_news_sentiment_rolling_dates.csv",
]

# Load and concatenate all CSV files
df_list = [pd.read_csv(file) for file in input_files]
df = pd.concat(df_list, ignore_index=True)

# Remove duplicate headlines
df = df.drop_duplicates(subset=["headline"])

# Save to a new CSV file
output_file = "Walt_Disney.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Combined CSV saved as: {output_file}")
