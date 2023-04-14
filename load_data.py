import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class PriceFeatureOnly(Dataset):
    def __init__(self, tickers, look_back=30, train=True):
        # for each ticker, we create a sequence of length L (look_back),
        # there will ba a lot of overlapping sequences
        # if a ticker contains 90 days of data, we can create 90 - L sequences of length L + 1
        # we will use the last day of each sequence as the target
        # so the target is the price of the next day based on the previous L days
        # the dataset input (N, L, D) -> (number_of_sequences, sequence_length, feature_dimension)
        # the dataset target (N, D) -> (number_of_sequences, feature_dimension)
        sequences = []  # (N, L, D)
        targets = []  # (N, D)
        scalers = []
        for ticker in tickers:
            df = pd.read_csv(f'./price/{ticker}.csv', index_col='Date', parse_dates=True)
            df = df[['Adj Close']]

            # (N, L, D) -> (number_of_sequence, sequence_length, feature_dimension)
            seq_data = []
            # (N, 1, D)
            seq_target = []

            # create all possible sequences of length look_back
            for index in range(len(df) - look_back):
                seq = df[index: index + look_back]
                target = df.iloc[index + look_back:index + look_back + 1]
                # scale the sequence data to be between -1 and 1, leave the target unscaled
                # we need different scaler for each sequence
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler = scaler.fit(seq.astype('float32'))
                seq = scaler.transform(seq.astype('float32'))  # Close index
                target = scaler.transform(target.astype('float32'))  # Close index
                seq_data.append(seq)
                seq_target.append(target)

                if not train:
                    # save the scaler for each sequence
                    scalers.append(scaler)

            seq_data = np.array(seq_data)
            seq_target = np.array(seq_target)

            # extend to sequences and targets, which contain all the sequences and targets for all tickers
            sequences.extend(seq_data)
            targets.extend(seq_target)

        # we combine all the sequences into one big tensor
        self.data = torch.Tensor(np.array(sequences))  # (N, L, D)
        self.target = torch.Tensor(np.array(targets))  # (N, 1, D)
        self.train = train
        self.look_back = look_back
        if not train:
            self.df = df[look_back:]
            self.scalers = scalers

    def invert_transform(self, data):
        new_data = np.array(data)
        for i in range(len(data)):
            new_data[i] = self.scalers[i].inverse_transform(data[i])
        new_data = new_data[:, :, 0]  # (N, 1, D) -> (N, 1)
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # X_train, y_train
        return self.data[idx], self.target[idx][:, 0]


class TransformerMultiFeaturesDataset(Dataset):
    def __init__(self, tickers, look_back=30, train=True):
        # for each ticker, we create a sequence of length L (look_back),
        # there will ba a lot of overlapping sequences
        # if a ticker contains 90 days of data, we can create 90 - L sequences of length L + 1
        # we will use the last day of each sequence as the target
        # so the target is the price of the next day based on the previous L days
        # the dataset input (N, L, D) -> (number_of_sequences, sequence_length, feature_dimension)
        # the dataset target (N, D) -> (number_of_sequences, feature_dimension)
        sequences = []  # (N, L, D)
        targets = []  # (N, D)
        scalers = []
        for ticker in tickers:
            df = pd.read_csv(f'./price/{ticker}.csv', index_col='Date', parse_dates=True)
            df = df[['Adj Close', 'High', 'Low', 'Volume']]

            # (N, L, D) -> (number_of_sequence, sequence_length, feature_dimension)
            seq_data = []
            # (N, 1, D)
            seq_target = []

            # create all possible sequences of length look_back
            for index in range(len(df) - look_back):
                seq = df[index: index + look_back]
                target = df.iloc[index + look_back:index + look_back + 1]
                # scale the sequence data to be between -1 and 1, leave the target unscaled
                # we need different scaler for each sequence
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler = scaler.fit(seq.astype('float32'))
                seq = scaler.transform(seq.astype('float32'))  # Close index
                target = scaler.transform(target.astype('float32'))  # Close index
                seq_data.append(seq)
                seq_target.append(target)

                if not train:
                    # save the scaler for each sequence
                    scalers.append(scaler)

            seq_data = np.array(seq_data)
            seq_target = np.array(seq_target)

            # extend to sequences and targets, which contain all the sequences and targets for all tickers
            sequences.extend(seq_data)
            targets.extend(seq_target)

        # we combine all the sequences into one big tensor
        self.data = torch.Tensor(np.array(sequences))  # (N, L, D)
        self.target = torch.Tensor(np.array(targets))  # (N, 1, D)
        self.train = train
        self.look_back = look_back
        if not train:
            self.df = df[look_back:]
            self.scalers = scalers

    def invert_transform(self, data):
        new_data = np.array(data)
        for i in range(len(data)):
            new_data[i] = self.scalers[i].inverse_transform(data[i])
        new_data = new_data[:, :, 0] # (N, 1, D) -> (N, 1)
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # X_train, y_train
        return self.data[idx], self.target[idx][:,0]


def get_data_loader(dataset, tickers, look_back=30, batch_size=32, shuffle=True, train=True):
    data = dataset(tickers, look_back, train)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

# train = get_data_loader(TransformerMultiFeaturesDataset, ['MSFT'], look_back=60, batch_size=32, shuffle=False)
# print(len(train))
# for it in train:
#     x, y = it
#     print(x.shape, y.shape)
#     break
