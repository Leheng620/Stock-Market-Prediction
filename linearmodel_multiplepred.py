import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, num_pred):
        super(Model, self).__init__()
        self.seq_len = 30
        self.pred_len = num_pred
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = 1
        self.Linear = nn.ModuleList()
        for i in range(self.channels):
            self.Linear.append(nn.Linear(self.seq_len,self.pred_len))


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
        for i in range(self.channels):
            output[:,:,i] = self.Linear[i](x[:,:,i])
        x = output
        return x # [Batch, Output length, Channel]
