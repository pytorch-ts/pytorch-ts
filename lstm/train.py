"""train
"""
import os
import numpy as np
import pandas as pd

import torch
import argparse
from lstm.model import LSTM, SMAPE
from lstm.data_processing import process_data


def cli_flag_argparser():
    """"Parser for command line flags."""
    flag_parser = argparse.ArgumentParser()
    flag_parser.add_argument('--data_path', type=str, default='../data/data.csv')
    flag_parser.add_argument('--checkpoint_path', type=str, default='')
    flag_parser.add_argument('--num_units', type=int, default=128)
    flag_parser.add_argument('--output_dim', type=int, default=1)
    flag_parser.add_argument('--num_layers', type=int, default=2)
    flag_parser.add_argument('--learning_rate', type=float, default=1e-3)
    flag_parser.add_argument('--num_epochs', type=int, default=40)
    flag_parser.add_argument('--batch_size', type=int, default=128)
    flag_parser.add_argument('--teacher_ratio', type=float, default=1.0)
    flag_parser.add_argument('--teacher_ratio_decay', type=float, default=0.995)
    flag_parser.add_argument('--forecast_length', type=int, default=5)
    flag_parser.add_argument('--window', type=int, default=30)
    flag_parser.add_argument('--dropout', type=float, default=0.2)
    flag_parser.add_argument('--validation_ratio', type=float, default=0.1)
    return flag_parser


def load_checkpoint(model, optimizer, file_name):
    """
    load model and optimizer
    :param model:
    :param optimizer:
    :param file_name:
    :return:
    """
    start_epoch = 0
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        optimizer.load_state_dict(checkpoint['optimizer'])
        # if use gpu should manually move to gpu
        print("=> loaded checkpoint '{}' (epoch {})".format(file_name,
                                                            start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(file_name))

    return model, optimizer, start_epoch


def _train(data, model, loss_fn, flags, teacher_ratio):
    train_x, target_y, feature_y = data
    new_batch_size = train_x.shape[0]
    model.batch_size = new_batch_size
    model.hidden = model.init_hidden()
    loss = 0.
    predictions = []
    for idx in range(flags.forecast_length):
        if idx == 0:
            inp = train_x
        else:
            inp = torch.cat([predictions[-1].reshape([-1, 1, 1]),
                            feature_y[:, idx-1:idx, :]], dim=2)
        y_true = target_y[:, idx]
        y_pred = model(inp)
        if np.random.random() < teacher_ratio:
            predictions.append(y_true)
        else:
            predictions.append(y_pred)
        loss += loss_fn(y_pred, y_true)
    avg_loss = loss / flags.forecast_length
    return avg_loss


def train(raw, flags):
    (loader, forecast_data, num_features, norm_dict) = process_data(
        raw, flags.forecast_length, flags.batch_size, flags.window,
        flags.validation_ratio)

    model = LSTM(num_features, flags.num_units, batch_size=flags.batch_size,
                 output_dim=flags.output_dim, num_layers=flags.num_layers,
                 batch_first=True, dropout=flags.dropout)

    loss_fn = SMAPE()

    opt = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)

    teacher_ratio = flags.teacher_ratio
    loss_history = []

    model, opt, start_epoch = load_checkpoint(model, opt, flags.checkpoint_path)
    if start_epoch >= flags.num_epochs:
        print('start_epoch is larger than num_epochs!')
    epoch = start_epoch
    for epoch in range(start_epoch, flags.num_epochs):
        for step, data in enumerate(loader):
            avg_loss = _train(data, model, loss_fn, flags, teacher_ratio)
            loss_history.append(avg_loss)
            opt.zero_grad()
            avg_loss.backward()
            opt.step()
            teacher_ratio *= flags.teacher_ratio_decay

        print('Epoch: %d' % epoch)
        print("Training Loss:%.3f" % avg_loss)
        print('Teacher_ratio: %.3f' % teacher_ratio)
        print()

    print('Model training completed and save at %s' % flags.checkpoint_path)

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': opt.state_dict()}
    torch.save(state, flags.checkpoint_path)
    return model, loss_history, forecast_data, num_features


# pylint: disable=too-many-locals
def infer(hparam, train_model, forecast_data, num_features, norm_list):
    """
    infer
    :param hparam:
    :param train_model:
    :param forecast_data:
    :param num_features:
    :param norm_list:
    :return:
    """
    model = LSTM(num_features, hparam.num_units, batch_size=hparam.batch_size,
                 output_dim=hparam.output_dim, num_layers=hparam.num_layers,
                 batch_first=True, dropout=0)
    model.load_state_dict(train_model.state_dict())
    model.eval()
    # predict
    results = []
    for index, data in enumerate(forecast_data):
        result = []
        predictions = []
        train_x, feature_y = map(torch.tensor, data)
        new_batch_size = train_x.shape[0]
        model.batch_size = new_batch_size
        mean, std = norm_list[index]
        model.hidden = model.init_hidden()
        for idx in range(hparam.forecast_length):
            if idx == 0:
                inp = train_x
            else:
                inp = torch.cat([predictions[-1].reshape([-1, 1, 1]),
                                 feature_y[:, idx - 1:idx, :]], dim=2)
            y_pred = model(inp)
            predictions.append(y_pred)
            pred = y_pred.detach().numpy() * std + mean
            result.append(pred[0])
        results.append(result)
    return results


def main(flags):
    raw = pd.read_csv(flags.data_path)
    train(raw, flags)


if __name__ == '__main__':
    FLAGS, _ = cli_flag_argparser().parse_known_args()
    main(FLAGS)
