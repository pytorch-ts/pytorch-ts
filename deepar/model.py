import torch
import torch.nn as nn


class DeepAR(nn.Module):
    def __init__(self, cov_dim, hidden_dim, num_class, embedding_dim, batch_first=True,
                 num_layers=2, dropout=0.):
        super(DeepAR, self).__init__()
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_class, embedding_dim)
        self.lstm = nn.LSTM(cov_dim + embedding_dim + 1, hidden_dim, num_layers,
                            batch_first=batch_first, dropout=dropout)

        self.mu = nn.Linear(num_layers * hidden_dim, 1)
        # softplus to make sure standard deviation is positive as paper described
        self.sigma_softplus = nn.Softplus()
        self.sigma_liner = nn.Linear(num_layers * hidden_dim, 1)
        self.hidden = None
        self.cell = None

    def forward(self, inp, ts_type_id):
        """

        :param inp: [batch_size, 1, 1 + cov_dim]
        :param ts_type_id: [batch_size, 1]
        :return:
        """
        batch_size = inp.shape[0]
        embedding = self.embedding(ts_type_id)
        lstm_input = torch.cat((inp, embedding.reshape([batch_size, 1, self.embedding_dim])), dim=2)
        output, (self.hidden, self.cell) = self.lstm(lstm_input, (self.hidden, self.cell))
        dist_hidden = self.hidden.permute(1, 2, 0).contiguous().view(batch_size, -1)
        mu = self.mu(dist_hidden)
        sigma = self.sigma_softplus(self.sigma_liner(dist_hidden))
        return mu.view(batch_size), sigma.view(batch_size)

    def init_hidden(self, batch_size):
        """
        init hidden state
        :return:
        """
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        self.cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim)


def gaussian_likelihood_loss(mu, sigma, labels):
    dist = torch.distributions.normal.Normal(mu, sigma)
    likelihood = dist.log_prob(labels)
    return -torch.mean(likelihood)
