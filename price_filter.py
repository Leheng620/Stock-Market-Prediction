import pandas as pd
import os

def price_filter():
    # Define the path to the directory containing the CSV files
    path = "price/"
    path_new = "price_filtered/"
    if not os.path.exists(path_new):
        os.mkdir(path_new)

    i = 1
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            # Load CSV file into a pandas DataFrame
            df = pd.read_csv(path+filename)

            # Compute percentage change in Adj_Close column and drop first row
            pct_change = abs(df["Adj Close"].pct_change())
            pct_change.loc[0] = 100.0 # keep the first day

            # Filter DataFrame to keep rows with pct_change greater than or equal to 0.005
            df_filtered = df[pct_change >= 0.005]
            
            # Save filtered DataFrame to a new CSV file with "_filtered" appended to the original filename
            df_filtered.to_csv(os.path.join(path_new, filename[:-4] + ".csv"), index=False)

            # output
            print(f"{i} Filtered {filename}: orginial #row is <{df.size}>, filtered #row is <{df_filtered.size}>")
            i += 1
    return


# def tweet_filter():
#     # Define the path to the directory containing the CSV files
#     path_price_filtered = "price_filtered/"
#     path_tweet = "tweet/"
#     path_tweet_new = "tweet_filtered/"
#     if not os.path.exists(path_tweet_new):
#         os.mkdir(path_tweet_new)

#     for filename in os.listdir(path_price_filtered):
#         if filename.endswith(".csv"):
#             for date in os.listdir(path_tweet+filename[:-4]+'/'):
#                 print(date)



if __name__ == "__main__":
    price_filter()
    # tweet_filter()