import torch
import torch.nn as nn


class DeepAR(nn.Module):
    def __init__(self, cov_dim, hidden_dim, num_class, embedding_dim,
                 batch_first=True, num_layers=2, dropout=0.):
        super(DeepAR, self).__init__()
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embedding = nn.Embedding(num_class, embedding_dim)
        self.lstm = nn.LSTM(cov_dim + embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=batch_first, dropout=dropout)

        self.mu = nn.Linear(num_layers * hidden_dim, 1)
        #softplus to make sure standard deviation is positive as paper described
        self.sigma = nn.Softplus(nn.Linear(num_layers * hidden_dim, 1))

    def forward(self, x, idx, hidden, cell):
        """

        :param x: [1, batch_size, 1+ cov_dim]
        :param idx: [1, batch_size]
        :param hidden: [num_layer, batch_size, lstm_hidden_dim]
        :param cell: [num_layer, batch_size, lstm_hidden_dim]
        :return:
        """
        batch_size = hidden.size[1]
        embedding = self.embedding(idx)
        lstm_input = torch.cat((x, embedding), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, hidden, cell)
        dist_hidden = hidden.permute(1, 2, 0).view(batch_size, -1)
        mu = self.mu(dist_hidden)
        sigma = self.sigma(dist_hidden)
        return mu.view(batch_size), sigma.view(batch_size), hidden, cell

    def init_hidden(self, batch_size):
        """
        init hidden state
        :return:
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


def gaussian_likelihood_loss(mu, sigma, labels):
    dist = torch.distributions.normal.Normal(mu, sigma)
    likelihood = dist.log_prob(labels)
    return -torch.mean(likelihood)
