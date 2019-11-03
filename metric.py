"""
time series metrics
"""
import torch
import torch.nn as nn


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


def accuracy(forecast, label):
    """
    average of (1 - abs(forecast -label)/label)
    :param forecast:
    :param label:
    :return:
    """
    error = torch.abs((label - forecast)/(label + 1e-3))
    acc = torch.max(torch.zeros_like(error), 1 - error)
    return torch.mean(acc)
