import math

import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device='cpu'):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.device = device

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, device=device)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x):
        x = x.to(self.device)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=False, device=self.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=False, device=self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, emb_dim, dim_feedforward=2048, nhead=8, num_encoder_layers=6, dropout=0.1, device='cpu'):
        super(EncoderBlock, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.device = device

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.multi_head = nn.MultiheadAttention(emb_dim, nhead, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, emb_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out1, _ = self.multi_head(x, x, x)
        out2 = self.layer_norm1(out1 + x)
        out2 = self.dropout(out2)
        out3 = self.feed_forward(out2)
        y = self.layer_norm2(out3 + out2)
        y = self.dropout(y)
        return y


class Encoder(nn.Module):
    def __init__(self, seq_len, emb_dim, dim_feedforward=2048, nhead=8, num_encoder_layers=6, dropout=0.1, device='cpu'):
        super(Encoder, self).__init__()
        # Hidden dimensions
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.device = device
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(seq_len, emb_dim, dim_feedforward, nhead, num_encoder_layers, dropout, device)
                for _ in range(num_encoder_layers)
            ])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x



class Transformer(nn.Module):
    def __init__(self, seq_len, emb_dim, hidden_dim=512, dim_feedforward=2048, nhead=8, num_encoder_layers=6, dropout=0.1, device='cpu'):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.device = device
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout

        self.encoder = Encoder(seq_len, emb_dim, dim_feedforward, nhead, num_encoder_layers, dropout, device)
        self.cross_attention = nn.MultiheadAttention(emb_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = x + self.position_encoding_sinusoid()
        last_time_step = x[:, -1:, :]
        x = self.encoder(x)
        x, _ = self.cross_attention(last_time_step, x, x)
        x = self.layer_norm(x + last_time_step)
        x = self.dropout_layer(x)
        x = x[:, 0, :]
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def position_encoding_sinusoid(self):
        seq_len, d_model = self.seq_len, self.emb_dim
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.to(self.device)

        return pe


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


# if __name__ == "__main__":
#     sineact = SineActivation(1, 8)
#     print(sineact(torch.Tensor([
#         [[7,1]],
#         [[1,2]]
#     ])).shape)

    # cosact = CosineActivation(1, 64)
    # print(cosact(torch.Tensor([[7]])).shape)
