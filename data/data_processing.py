import os
import json
import csv
from datetime import datetime


import json

def create_time_window_data(data, time_window):
    time_window_data = []
    for i in range(len(data) - time_window + 1):
        window_data = {
            'name': data[i]['name'],
            'date': data[i]['date'],
            'open': [float(d['open']) for d in data[i:i+time_window-1]],
            'high': [float(d['high']) for d in data[i:i+time_window-1]],
            'low': [float(d['low']) for d in data[i:i+time_window-1]],
            'close': [float(d['close']) for d in data[i:i+time_window-1]],
            'volume': [float(d['volume']) for d in data[i:i+time_window-1]],
            'adj_close': [float(d['adj_close']) for d in data[i:i+time_window-1]],
            'text': [d['text'] for d in data[i:i+time_window-1]],
            'label': 1 if data[i+time_window-1]['close'] >= data[i+time_window-2]['close'] else 0
        }
        time_window_data.append(window_data)
    return time_window_data




def preprocess(text):
    new_text = []
    for t in text:
        t = '@user' if t == "AT_USER" else t
        new_text.append(t)
    return " ".join(new_text)

def read_stock_prices(stock_name):
    file_path = os.path.join("price", f"{stock_name}.csv")
    stock_data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stock_data.append(row)
    return stock_data

def read_twitter_posts(stock_name):
    folder_path = os.path.join("tweet", stock_name)
    twitter_data = {}
    for filename in os.listdir(folder_path):
        date = filename.split(".")[0]
        with open(os.path.join(folder_path, filename), "r") as f:
            lines = f.readlines()
            tweets = [json.loads(line) for line in lines]
            twitter_data[date] = [preprocess(tweet["text"]) for tweet in tweets]
            twitter_data[date] = list(set(twitter_data[date]))
            twitter_data[date] = " ".join(twitter_data[date])
    return twitter_data

def align_stock_and_twitter_data(stock_name):
    stock_data = read_stock_prices(stock_name)
    twitter_data = read_twitter_posts(stock_name)
    
    aligned_data = []

    for row in stock_data:
        date = row["Date"]
        twitter_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")

        if twitter_date in twitter_data:
            data = {
                "name": stock_name,
                "date": date,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "adj_close": float(row["Adj Close"]),
                "text": twitter_data[twitter_date],
            }
            aligned_data.append(data)

    return aligned_data

def save_aligned_data(stock_name, aligned_data):
    output_path = f"./tweet_price/{stock_name}aligned_data.json"
    with open(output_path, "w") as f:
        json.dump(aligned_data, f, indent=4)


def balance_labels(all_data):
    count_0, count_1 = 0, 0

    # Count the number of 0s and 1s in the labels
    for data in all_data:
        if data['label'] == 0:
            count_0 += 1
        else:
            count_1 += 1

    # Calculate the minimum count to balance the dataset
    min_count = min(count_0, count_1)
    print("Count of 0s: ", count_0)
    print("Count of 1s: ", count_1)
    print("Minimum count: ", min_count)
          

    # Create a new list to store the balanced data
    balanced_data = []
    count_0, count_1 = 0, 0

    # Add instances to the balanced data until both labels reach the min_count
    for window_data in time_window_data:
        if window_data['label'] == 0 and count_0 < min_count:
            balanced_data.append(window_data)
            count_0 += 1
        elif window_data['label'] == 1 and count_1 < min_count:
            balanced_data.append(window_data)
            count_1 += 1

        # Break the loop if we have enough instances of both labels
        if count_0 == min_count and count_1 == min_count:
            break

    return balanced_data

        

all = list()
for item in os.listdir('./tweet/'):
    stock_name = item
    print(stock_name)
    aligned_data = align_stock_and_twitter_data(stock_name)
    save_aligned_data("test",aligned_data)
    time_window = 5  # Change this value according to your needs
    time_window_data = create_time_window_data(aligned_data, time_window)
    all += (time_window_data)
balanced_data = balance_labels(all)

save_aligned_data(stock_name, balanced_data)
