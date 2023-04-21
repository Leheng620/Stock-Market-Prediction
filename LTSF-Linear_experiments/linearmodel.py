import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Use this for simple linear method (comment out if using Nlinear)
class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 4
        self.pred_len = 1
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = 15
        self.Linear = nn.ModuleList()
        for i in range(self.channels):
            self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        self.fc = nn.Linear(self.channels, 1)


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
        for i in range(self.channels):
            output[:,:,i] = self.Linear[i](x[:,:,i])
        output = self.fc(output)
        x = output
        return x # [Batch, Output length, Channel]




## Use this for Normalized linear method (comment out if using linear)
# class Model(nn.Module):
#     """
#     Normalization-Linear
#     """
#     def __init__(self):
#         super(Model, self).__init__()
#         self.seq_len = 29
#         self.pred_len = 1
        
#         # Use this line if you want to visualize the weights
#         # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         self.channels = 15
#         self.Linear = nn.ModuleList()
#         for i in range(self.channels):
#             self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         self.fc = nn.Linear(self.channels, 1)

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seq_last = x[:,-1:,:].detach()
#         x = x - seq_last
#         output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
#         for i in range(self.channels):
#             output[:,:,i] = self.Linear[i](x[:,:,i])
#         x = output
#         x = x + seq_last
#         x = self.fc(x)
#         return x # [Batch, Output length, Channel]