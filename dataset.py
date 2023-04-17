
import os
import torch
import json
import datetime
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

def is_weekday(date_string):
    date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    day_of_week = date_obj.weekday()
    return day_of_week < 5


class PriceTweetDataset(Dataset):
    def __init__(self, tickers, look_back=30, train=True):
        # for each ticker, we create a sequence of length L (look_back),
        # there will ba a lot of overlapping sequences
        # if a ticker contains 90 days of data, we can create 90 - L sequences of length L
        # we will use the last day of each sequence as the target
        # so the target is the price of the next day based on the previous L-1 days
        # the dataset input (N, L-1, D) -> (number_of_sequences, sequence_length, feature_dimension)
        # the dataset target (N, D) -> (number_of_sequences, feature_dimension)
        # the feature dimension is 1, because we only use the price of the stock as the feature


        sequences = [] # (N, L-1, D)
        targets = [] # (N, D)
        # Load the BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        
        for ticker in tickers:
            # (N, L-1, D) -> (number_of_sequence, sequence_length, feature_dimension)
            seq_data = []
            # deal with price data
            df = pd.read_csv(f'./price/{ticker}.csv', index_col='Date', parse_dates=True)
            df = df[['Adj Close']]
            # scale the price data to be between -1 and 1
            # we need different scaler for each ticker/stock
            scaler = MinMaxScaler(feature_range=(-1, 1))
            minmax = scaler.fit(df.astype('float32'))
            df_log = minmax.transform(df.astype('float32'))  # Close index

            # create all possible sequences of length look_back
            for index in range(len(df_log) - look_back):
                seq_data.append(df_log[index: index + look_back])
            # deal with tweet data
            # (N, L-1, max_length) -> (number_of_sequence, sequence_length, 512)
            # one day may have multiple tweets, so we need to concatenate them
            # for weekend, we just concatenate all the tweets together
            data = []
            dates = os.listdir(f"./tweet/{ticker}/")
            for i in range(len(dates)):
                data.append([])
                if is_weekday(dates[i]):
                    with open(f'tweet/{ticker}/{dates[i]}', 'r') as file:
                        for line in file:
                            json_line = json.loads(line.strip())
                            data[i] += json_line["text"]
                        data[i] = ' '.join(data[i])
                else:
                    data[i] = ' '.join(data[i])

                tokenized_data = [tokenizer.encode(text[0], padding='max_length', max_length=512, truncation=True) for text in data]

                # create all possible sequences of length look_back
                for index in range(len(tokenized_data) - look_back):
                    seq_data.append(tokenized_data[index: index + look_back])
            
            seq_data = np.array(seq_data)
            
            target = seq_data[:, -1, :]
            seq_data = seq_data[:, :-1, :]
            # extend to sequences and targets, which contain all the sequences and targets for all tickers
            sequences.extend(seq_data)
            targets.extend(target)

        # we combine all the sequences into one big tensor
        self.data = torch.Tensor(np.array(sequences)) # (N, L-1, D)
        self.target = torch.Tensor(np.array(targets)) # (N, D)
        print("\n******************************************************************************")
        print("The dimension of training (*{}*) data is {}".format(priceORtweet, self.data.size()))
        print("The dimension of testing (*{}*) data is {}".format(priceORtweet, self.target.size()))

        
        # (N, L-1, D)   (N, L-1, 512)
        # -> (N, 1)

        if not train:
            # the test set should only have one ticker, so the scaler can be saved
            # we use this scaler to invert transform the predictions back to price
            # df is the dataframe of the ticker, whose index is the date and the column is the price
            self.scaler = scaler
            self.df = df

    def invert_transform(self, data):
        return self.scaler.inverse_transform(data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def get_data_loader(dataset, tickers, priceOrtweet, look_back=30, batch_size=32, shuffle=True, train=True):
    data = dataset(tickers, priceOrtweet, look_back, train)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

# train = get_data_loader(PriceTweetDataset, ['AAPL', 'AMZN'], priceOrtweet='tweet', look_back=30, batch_size=32, shuffle=True)
# print(len(train))



if __name__ == "__main__":
    # test the dataset
    data = PriceTweetDataset(['AAPL', 'AMZN'], look_back = 10, train = True)
    print(data[0])