import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_head, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model = model_dim, num_head=num_head, num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_in = nn.Linear(input_dim, model_dim)
        self.fc_out = nn.Linear(model_dim,1)
        self.positional_encoding = nn.Parameter(torch.zeros(1,1000, model_dim)) 
    def forward(self, src):
        src = self.fc_in(src) + self.positional_encoding[:, :src.size(1), :]
        transformer_output = self.transformer(src, src)
        output = self.fc_out(transformer_output[:,-1,:])
        return output