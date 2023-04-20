import torch
import torch.nn as nn

class RobertaLSTM(nn.Module):
    def __init__(self, transformer_model, time_window):
        super(RobertaLSTM, self).__init__()
        self.transformer_model = transformer_model
        self.lstm = nn.LSTM(768 + 2, 64, batch_first=True)  # 768: RoBERTa base hidden size, 2: volume & adj_close
        self.linear = nn.Linear(64, 1)
        self.time_window = time_window

    def forward(self, input_ids, attention_mask, prices, volumes):
        
        sentiment_scores_list = []
        for i in range(self.time_window-1):
            outputs = self.transformer_model(input_ids[:, i, :], attention_mask=attention_mask[:, i, :])
            logits = outputs.logits
            sentiment_scores = torch.softmax(logits, dim=-1)[:, 1]  # Positive sentiment scores
            sentiment_scores_list.append(sentiment_scores.unsqueeze(-1))
        
        sentiment_scores_combined = torch.stack(sentiment_scores_list, dim=1)
        features = torch.cat([sentiment_scores_combined, prices, volumes], dim=-1)
        
        lstm_out, _ = self.lstm(features)
        out = self.linear(lstm_out)
        return out.squeeze(-1)
    






