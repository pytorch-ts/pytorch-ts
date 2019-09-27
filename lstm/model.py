"""Pytorch lstm
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    lstm
    """

    def __init__(self, input_dim, hidden_dim, batch_first=True,
                 output_dim=1, num_layers=2, dropout=0.):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Define the lstm layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            batch_first=batch_first, dropout=dropout)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.hidden = None

    def init_hidden(self, batch_size):
        """
        init hidden state
        :return:
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
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


class SMAPE(nn.Module):
    """Compute symmetric mean absolute percentage error for Tensors."""
    def __init__(self, epsilon=0.1):
        super(SMAPE, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        forward
        :param y_pred:
        :param y_true:
        :return:
        """
        summ = torch.max(torch.abs(y_true) + torch.abs(y_pred) + self.epsilon,
                         torch.tensor(0.5 + self.epsilon))
        error = torch.abs(y_true - y_pred) / summ * 2.0
        loss = torch.mean(error)
        return loss
