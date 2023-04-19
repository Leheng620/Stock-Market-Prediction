import pandas as pd
import os

# Define the path to the directory containing the CSV files
path = "price/"
path_new = "price_filtered/"
if not os.path.exists(path_new):
    os.mkdir(path_new)
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(path+filename)

        print(df)
        # Compute percentage change in Adj_Close column and drop first row
        pct_change = df["Open"].pct_change().dropna()
        
        # Filter DataFrame to keep rows with pct_change greater than or equal to 0.005
        df_filtered = df[pct_change >= 0.005]
        
        # Save filtered DataFrame to a new CSV file with "_filtered" appended to the original filename
        df_filtered.to_csv(os.path.join(path_new, filename[:-4] + "_filtered.csv"), index=False)