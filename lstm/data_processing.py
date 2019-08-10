"""
data processing
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from common.features import last_year_value, is_weekends, is_workdays, \
    is_holidays


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    convert series data to wide table, including previous input and target
    :param data:
    :param n_in:
    :param n_out:
    :param dropnan:
    :return:
    """
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s_(t-%d)' % (j, i)) for j in data.columns]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % j) for j in data.columns]
        else:
            names += [('%s_(t+%d)' % (j, i)) for j in data.columns]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg = agg.astype(np.float32)
    return agg


class LSTMDataSet(Dataset):
    """
    data set
    """
    def __init__(self, train_x, target_y, feature_y):
        self.train_x = train_x
        self.target_y = target_y
        self.feature_y = feature_y

    def __len__(self):
        return self.train_x.shape[0]

    def __getitem__(self, idx):
        return (self.train_x[idx, :, :], self.target_y[idx, :],
                self.feature_y[idx, :, :])


def process_data(raw, forecast_length, batch_size=128, window=30,
                 validation_ratio=0.1):
    # TODO: add validation
    data_list = []
    forecast_data_list = []
    num_features = 5
    norm_dict = {}
    raw['date'] = pd.to_datetime(raw['date'])
    end_date = max(raw['date'])
    future_df = pd.DataFrame({'date': [(end_date + pd.Timedelta(days=i))
                                              for i in
                                              range(1, forecast_length + 1)],
                              'y': np.zeros(forecast_length)})
    ts_types = raw['type'].unique()
    for type in ts_types:
        data = raw.loc[(raw.type == type)]
        y_mean, y_std = np.mean(data.y), np.std(data.y)
        norm_dict['type'] = [y_mean, y_std]
        data['y'] = (data['y'] - y_mean) / y_std
        data = data[['date', 'y']]

        data = pd.concat([data, future_df], ignore_index=True)

        data['is_weekend'] = is_weekends(data.date)
        data['is_workday'] = is_workdays(data.date)
        data['is_holiday'] = is_holidays(data.date)
        data['last_year_value'] = last_year_value(data, 'date', 'y')

        del data['date']
        train = series_to_supervised(data, window, forecast_length)
        n_obs = window * num_features

        train_X, train_y = train.iloc[:, :n_obs], train.iloc[:, n_obs:]
        train_X = train_X.values.reshape(train_X.shape[0], window, num_features)
        targe_y = train_y[[col for col in train_y.columns if
                           col.startswith('y')]]
        targe_y = targe_y.values.reshape(train_X.shape[0], forecast_length)
        feature_y = train_y[[col for col in train_y.columns if
                             not col.startswith('y')]]
        feature_y = feature_y.values.reshape(train_X.shape[0], forecast_length,
                                             num_features - 1)
        data_list.append((train_X[:-forecast_length, :, :],
                          targe_y[:-forecast_length, :],
                          feature_y[:-forecast_length, :, :]))
        forecast_data_list.append((train_X[-1:, :, :], feature_y[-1:, :, :]))

    train_x, target_y, feature_y = map(np.concatenate, zip(*data_list))
    dataset = LSTMDataSet(train_x, target_y, feature_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, forecast_data_list, num_features, norm_dict
