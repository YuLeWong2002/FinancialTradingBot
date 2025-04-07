import os
import pandas as pd

# Define the correct folder name
input_dir = 'stock_data'

# Loop through each CSV file in the folder
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path)

        # Check and rename the first column if unnamed
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df.to_csv(path, index=False)
            print(f"✅ Fixed: {filename}")
        else:
            print(f"✅ Already good: {filename}")