"""Pytorch lstm
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    lstm
    """

    def __init__(self, cov_dim, hidden_dim, batch_first=True, output_dim=1, num_layers=2,
                 dropout=0.):
        super(LSTM, self).__init__()
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Define the lstm layer
        self.lstm = nn.LSTM(self.cov_dim + 1, self.hidden_dim, self.num_layers,
                            batch_first=batch_first, dropout=dropout)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.hidden = None

    def init_hidden(self, batch_size):
        """
        init hidden state
        :return:
        """
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                       torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inp):
        """
        forward
        :param input:
        :return:
        """
        lstm_out, self.hidden = self.lstm(inp, self.hidden)
        out = self.linear(lstm_out[:, -1, :])
        return out.view(-1)
