"""
data processing
"""
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from common.features import last_year_value, is_workdays, is_holidays
from scipy import stats
from sklearn.preprocessing import LabelEncoder


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    convert series data to wide table, including previous input and target,
    more detailed can be found at https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
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


class TsDataSet(Dataset):
    """
    data set
    """

    def __init__(self, label_x, feature_x, label_y, feature_y, scale, ts_type):
        self.label_x = label_x
        self.feature_x = feature_x
        self.label_y = label_y
        self.feature_y = feature_y
        self.scale = scale
        self.ts_type = ts_type

    def __len__(self):
        return self.label_x.shape[0]

    def __getitem__(self, idx):
        return (self.label_x[idx, :], self.feature_x[idx, :, :], self.label_y[idx, :],
                self.feature_y[idx, :, :], self.scale[idx], self.ts_type[idx])


class DataProcessor:
    """
    process training and forecast data
    """

    def __init__(self, forecast_length, batch_size, window):
        """

        :param forecast_length:
        :param batch_size:
        :param window:
        """
        self.forecast_length = forecast_length
        self.batch_size = batch_size
        self.window = window
        self.label_encoder = LabelEncoder()
        self.num_features = 6

    def _get_x_y(self, df):
        data = series_to_supervised(df, self.window, self.forecast_length)
        n_obs = self.window * (self.num_features + 1)

        data_x, data_y = data.iloc[:, :n_obs], data.iloc[:, n_obs:]
        feature_x = data_x[[col for col in data_x.columns if not col.startswith('y')]]
        feature_x = feature_x.values.reshape(data_x.shape[0], self.window, self.num_features)
        label_x = data_x[[col for col in data_x.columns if col.startswith('y')]]
        label_x = label_x.values.reshape(data_x.shape[0], self.window)

        label_y = data_y[[col for col in data_y.columns if col.startswith('y')]]
        label_y = label_y.values.reshape(data_x.shape[0], self.forecast_length)
        feature_y = data_y[[col for col in data_y.columns if not col.startswith('y')]]
        feature_y = feature_y.values.reshape(data_x.shape[0], self.forecast_length,
                                             self.num_features)
        return feature_x, label_x, feature_y, label_y

    def get_train_test_data(self, raw, if_scale=True, val_ratio=0.1):
        """

        :param raw: pandas dataframe, with column 'date','y','type'
        :param val_ratio:
        :return:
        """
        train_data_list = []
        train_scale_list = []
        val_data_list = []
        val_scale_list = []
        train_type_list = []
        val_type_list = []

        raw['date'] = pd.to_datetime(raw['date'])
        ts_types = sorted(raw['type'].unique())
        self.label_encoder.fit(ts_types)

        self.start = raw['date'].min()
        raw['time_index'] = (raw['date'] - self.start) / pd.Timedelta(days=365)

        for type in ts_types:
            df = raw.loc[(raw.type == type)].copy()
            df = df.sort_values(by='date')
            df.reset_index(drop=True, inplace=True)
            # features
            df['is_workday'] = is_workdays(df.date)
            df['is_holiday'] = is_holidays(df.date)
            last_year = last_year_value(df, 'date', 'y')
            if np.sum(last_year == 0.) == len(last_year):
                df['last_year_value'] = last_year
            else:
                df['last_year_value'] = stats.zscore(last_year)
            df['weekday'] = stats.zscore(df['date'].dt.weekday)
            df['month'] = stats.zscore(df['date'].dt.month)

            del df['type']
            del df['date']

            feature_x, label_x, feature_y, label_y = self._get_x_y(df)

            val_length = max(1, int(label_x.shape[0] * val_ratio))

            # scale value as described in deepar paper
            if if_scale:
                scale = 1 + np.mean(label_x, axis=1)
            else:
                scale = 1 + np.zeros(label_x.shape[0], dtype=np.float32)
            label_x = label_x / scale[:, np.newaxis]
            label_y = label_y / scale[:, np.newaxis]
            train_scale = scale[:-val_length]
            val_scale = scale[-val_length:]
            train_scale_list.append(train_scale)
            val_scale_list.append(val_scale)

            # train data
            train_data_list.append((label_x[:-val_length, :], feature_x[:-val_length, :, :],
                                    label_y[:-val_length, :], feature_y[:-val_length, :, :]))

            # validation data
            val_data_list.append((label_x[-val_length:, :], feature_x[-val_length:, :, :],
                                  label_y[-val_length:, :], feature_y[-val_length:, :, :]))

            train_type_list.append(
                self.label_encoder.transform([type] * label_x[:-val_length, :].shape[0]))
            val_type_list.append(self.label_encoder.transform([type] * val_length))

        train_data = list(map(np.concatenate, zip(*train_data_list)))
        train_scale = np.concatenate(train_scale_list)
        train_types = np.concatenate(train_type_list)

        train_dataset = TsDataSet(*(train_data + [train_scale, train_types]))
        sampler = WeightedRandomSampler(np.abs(train_scale), train_scale.shape[0])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)

        val_data = list(map(np.concatenate, zip(*val_data_list)))
        val_scale = np.concatenate(val_scale_list)
        val_types = np.concatenate(val_type_list)

        val_dataset = TsDataSet(*(val_data + [val_scale, val_types]))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader

    def get_forecast_data(self, raw):
        forecast_data_list = []
        scale_list = []
        type_list = []
        raw['date'] = pd.to_datetime(raw['date'])
        ts_types = sorted(raw['type'].unique())
        for type in ts_types:
            df = raw.loc[(raw.type == type)].copy()
            end_date = max(df.date)
            df = df.sort_values(by='date')
            future_df = pd.DataFrame({'date': [(end_date + pd.Timedelta(days=i)) for i in
                                               range(1, self.forecast_length + 1)],
                                      'y': np.zeros(self.forecast_length),
                                      'type': [type] * self.forecast_length})
            df = pd.concat([df, future_df], axis=0, sort=False)
            # features
            df['is_workday'] = is_workdays(df.date)
            df['is_holiday'] = is_holidays(df.date)
            df['last_year_value'] = stats.zscore(last_year_value(df, 'date', 'y'))
            df['weekday'] = stats.zscore(df['date'].dt.weekday)
            df['month'] = stats.zscore(df['date'].dt.month)
            del df['type']
            del df['date']
            feature_x, label_x, feature_y, label_y = self._get_x_y(df)
            # scale value as described in deepar paper
            forecast_feature_x = feature_x[-1:, :, :]
            forecast_label_x = label_x[-1:, :]
            forecast_feature_y = feature_y[-1:, :, :]
            # it's fake data, just for using the Dataset
            forecast_label_y = label_y[-1:, :]
            assert np.sum(forecast_label_y) == 0

            scale = 1 + np.mean(forecast_label_x, axis=1)
            forecast_label_x = forecast_label_x / scale[:, np.newaxis]
            forecast_data_list.append(
                [forecast_label_x, forecast_feature_x, forecast_label_y, forecast_feature_y])
            scale_list.append(scale)
            type_list.append(self.label_encoder.transform([type]))
        data = list(map(np.concatenate, zip(*forecast_data_list)))
        scale = np.concatenate(scale_list)
        types = np.concatenate(type_list)
        dataset = TsDataSet(*(data + [scale, types]))
        loader = DataLoader(dataset, batch_size=1)
        return loader, ts_types

    def save(self, file_name):
        np.save(file_name + '/classes.npy', self.label_encoder.classes_)
        with open(file_name + '/data_process_params.json', 'w') as f:
            json.dump({'forecast_length': self.forecast_length, 'batch_size': self.batch_size,
                       'window': self.window}, f)

    @staticmethod
    def load(file_name):
        with open(file_name + '/data_process_params.json', 'r') as f:
            param = json.load(f)
        data_processor = DataProcessor(**param)
        data_processor.label_encoder.classes_ = np.load(file_name + '/classes.npy')
        return data_processor
