"""train
"""
import os
import numpy as np
import pandas as pd

import torch
import argparse
from lstm.model import LSTM
from metric import SMAPE, accuracy
from data_processing import DataProcessor


def cli_flag_argparser():
    """"Parser for command line flags."""
    flag_parser = argparse.ArgumentParser()
    flag_parser.add_argument('--data_path', type=str, default='../data/data.csv')
    flag_parser.add_argument('--mode', type=str, default='train')
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


def load_checkpoint(file_name, model, optimizer=None):
    """
    load model and optimizer
    :param model:
    :param optimizer:
    :param file_name:
    :return:
    """
    start_epoch = 0
    if os.path.isfile(file_name + '/model.pt'):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name + '/model.pt')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        # if use gpu should manually move to gpu
        print("=> loaded checkpoint '{}' (epoch {})".format(file_name, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(file_name))

    return model, optimizer, start_epoch


def _forward(data, model, loss_fn, window, forecast_length, training=True, teacher_ratio=1):
    outputs = []
    label_x, feature_x, label_y, feature_y, _, _ = data
    batch_size = label_x.shape[0]
    model.init_hidden(batch_size)
    # concat the true value of day(t-1) and the features of day(t) to forecast day(t)
    inp = torch.cat(
        [label_x[:, :-1].reshape(label_x.shape[0], label_x.shape[1] - 1, 1), feature_x[:, 1:, :]],
        dim=2)
    # no need to iterate the first day
    for time_step in range(window - 1):
        output = model(inp[:, time_step:time_step + 1, :])
        outputs.append(output)
    for idx in range(forecast_length):
        if idx == 0:
            inp = torch.cat([label_x[:, -1:].reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]],
                            dim=2)
        else:
            if training:
                if np.random.random() < teacher_ratio:
                    inp = torch.cat(
                        [label_y[:, idx - 1].reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]],
                        dim=2)
                else:
                    inp = torch.cat([output.reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]],
                                    dim=2)
            else:
                inp = torch.cat([output.reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]], dim=2)

        output = model(inp)
        outputs.append(output)
    outputs = torch.stack(outputs, 1)
    loss = loss_fn(outputs, torch.cat([label_x[:, 1:], label_y], 1))

    avg_acc = accuracy(outputs[:, -forecast_length:], label_y)
    return loss, outputs, avg_acc


def _infer(data, model, window, forecast_length):
    outputs = []
    label_x, feature_x, _, feature_y, _, _ = data
    batch_size = label_x.shape[0]
    model.init_hidden(batch_size)
    # concat the true value of day(t-1) and the features of day(t) to forecast day(t)
    inp = torch.cat(
        [label_x[:, :-1].reshape(label_x.shape[0], label_x.shape[1] - 1, 1), feature_x[:, 1:, :]],
        dim=2)
    # no need to iterate the first day
    for time_step in range(window - 1):
        output = model(inp[:, time_step:time_step + 1, :])
        outputs.append(output)
    for idx in range(forecast_length):
        if idx == 0:
            inp = torch.cat([label_x[:, -1:].reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]],
                            dim=2)
        else:
            inp = torch.cat([output.reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]], dim=2)

        output = model(inp)
        outputs.append(output)
    outputs = torch.stack(outputs, 1)
    return outputs


def evaluate(val_loader, model, loss_fn, window, forecast_length):
    losses = []
    accs = []
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            loss, output, acc = _forward(data, model, loss_fn, window, forecast_length, False)
            losses.append(loss)
            accs.append(acc)
    return np.mean(losses), np.mean(accs)


def train(raw, flags):
    data_processor = DataProcessor(flags.forecast_length, flags.batch_size, flags.window)
    train_loader, val_loader = data_processor.get_train_test_data(raw, flags.validation_ratio)

    model = LSTM(data_processor.num_features, flags.num_units, output_dim=flags.output_dim,
                 num_layers=flags.num_layers, batch_first=True, dropout=flags.dropout)

    loss_fn = SMAPE()

    opt = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)

    teacher_ratio = flags.teacher_ratio
    loss_history = []

    model, opt, start_epoch = load_checkpoint(flags.checkpoint_path, model, opt)
    if start_epoch >= flags.num_epochs:
        print('start_epoch is larger than num_epochs!')
    epoch = start_epoch
    # TODO: add early stop
    for epoch in range(start_epoch, flags.num_epochs):
        for step, data in enumerate(train_loader):
            avg_loss, _, acc = _forward(data, model, loss_fn, flags.window, flags.forecast_length,
                                        True, teacher_ratio)
            loss_history.append(avg_loss)
            opt.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            teacher_ratio *= flags.teacher_ratio_decay
        val_loss, val_acc = evaluate(val_loader, model, loss_fn, flags.window,
                                     flags.forecast_length)
        print('Epoch: %d' % epoch)
        print("Training Loss:%.3f" % avg_loss)
        print('Training Avg Accuracy:%.3f' % acc)
        print("Validation Loss:%.3f" % val_loss)
        print("Validation Accuracy:%.3f" % val_acc)
        print('Teacher_ratio: %.3f' % teacher_ratio)
        print('Gradients:%.3f' % torch.mean(
            (torch.stack([torch.mean(torch.abs(p.grad)) for p in model.parameters()], 0))))
        print()

    print('Model training completed and save at %s' % flags.checkpoint_path)
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}
    if not os.path.exists(flags.checkpoint_path):
        os.mkdir(flags.checkpoint_path)
    torch.save(state, flags.checkpoint_path + '/model.pt')
    data_processor.save(flags.checkpoint_path)
    return model, loss_history


# pylint: disable=too-many-locals
def infer(raw, flags):
    """

    :param raw:
    :param flags:
    :return:
    """
    data_processor = DataProcessor.load(flags.checkpoint_path)
    model = LSTM(data_processor.num_features, flags.num_units, output_dim=flags.output_dim,
                 num_layers=flags.num_layers, batch_first=True, dropout=flags.dropout)
    model, _, _ = load_checkpoint(flags.checkpoint_path, model, None)
    model.eval()
    # predict
    results = {}
    loader, ts_types = data_processor.get_forecast_data(raw)
    with torch.no_grad():
        for type, data in zip(ts_types, loader):
            scale = data[4]
            _, outputs = _infer(data, model, flags.window, flags.forecast_length)
            results[type] = [(output * scale).detach().numpy()[0] for output in outputs]
    print(results)
    return results


def main(flags):
    raw = pd.read_csv(flags.data_path)
    if flags.mode == 'train':
        train(raw, flags)
    elif flags.mode == 'infer':
        infer(raw, flags)
    else:
        raise ValueError('argument --mode must be train or infer')


if __name__ == '__main__':
    FLAGS, _ = cli_flag_argparser().parse_known_args()
    main(FLAGS)
