"""train
"""
import os
import numpy as np
import pandas as pd

import torch
import argparse
from lstm.model import LSTM, SMAPE
from data_processing import DataProcessor


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
    flag_parser.add_argument('--teacher_ratio_decay', type=float, default=1.0)
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
        print("=> loaded checkpoint '{}' (epoch {})".format(file_name, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(file_name))

    return model, optimizer, start_epoch


def _train(data, model, loss_fn, window, forecast_length, teacher_ratio):
    label_x, feature_x, label_y, feature_y, scale = data
    new_batch_size = label_x.shape[0]
    model.batch_size = new_batch_size
    model.hidden = model.init_hidden(new_batch_size)
    loss = 0.
    inp = torch.cat([label_x.reshape(label_x.shape[0], label_x.shape[1], 1), feature_x], dim=2)
    for time_step in range(window):
        output = model(inp[:, time_step:time_step + 1, :])

    for idx in range(forecast_length):
        y_true = label_y[:, idx]
        if np.random.random() < teacher_ratio:
            inp = torch.cat([y_true.reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]], dim=2)
        else:
            inp = torch.cat([output.reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]], dim=2)

        output = model(inp)
        loss += loss_fn(output, y_true)
    avg_loss = loss / forecast_length
    return avg_loss


def evaluate(val_loader, model, loss_fn, window, forecast_length):
    losses = []
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            loss = _train(data, model, loss_fn, window, forecast_length, 1.)
            losses.append(loss)
    return np.mean(losses)


def train(raw, flags):
    data_processor = DataProcessor(flags.forecast_length, flags.batch_size, flags.window, False)
    train_loader, val_loader = data_processor.process_train_test_data(raw, flags.validation_ratio)

    model = LSTM(data_processor.num_features + 1, flags.num_units, output_dim=flags.output_dim,
                 num_layers=flags.num_layers, batch_first=True, dropout=flags.dropout)

    loss_fn = SMAPE()

    opt = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)

    teacher_ratio = flags.teacher_ratio
    loss_history = []

    model, opt, start_epoch = load_checkpoint(model, opt, flags.checkpoint_path)
    if start_epoch >= flags.num_epochs:
        print('start_epoch is larger than num_epochs!')
    epoch = start_epoch
    # TODO: add early stop
    for epoch in range(start_epoch, flags.num_epochs):
        for step, data in enumerate(train_loader):
            avg_loss = _train(data, model, loss_fn, flags.window, flags.forecast_length,
                              teacher_ratio)
            loss_history.append(avg_loss)
            opt.zero_grad()
            avg_loss.backward()
            opt.step()
            teacher_ratio *= flags.teacher_ratio_decay
        validation_loss = evaluate(val_loader, model, loss_fn, flags.window, flags.forecast_length)
        print('Epoch: %d' % epoch)
        print("Training Loss:%.3f" % avg_loss)
        print("Validation Loss:%.3f" % validation_loss)
        print('Teacher_ratio: %.3f' % teacher_ratio)
        print()

    print('Model training completed and save at %s' % flags.checkpoint_path)

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}
    torch.save(state, flags.checkpoint_path)
    return model, loss_history


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
    # TODO: update infer
    model = LSTM(num_features, hparam.num_units, output_dim=hparam.output_dim,
                 num_layers=hparam.num_layers, batch_first=True, dropout=0)
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
        model.hidden = model.init_hidden(new_batch_size)
        for idx in range(hparam.forecast_length):
            if idx == 0:
                inp = train_x
            else:
                inp = torch.cat([predictions[-1].reshape([-1, 1, 1]), feature_y[:, idx - 1:idx, :]],
                                dim=2)
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
