"""train
"""
import os
import numpy as np
import pandas as pd

import torch
import argparse
from seq2seq.model import EncoderRNN, DecoderRNN
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
    flag_parser.add_argument('--teacher_ratio_decay', type=float, default=1.0)
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


def _forward(data, model, loss_fn, window, forecast_length, training, teacher_ratio=1):
    encoder, decoder = model
    outputs = []
    label_x, feature_x, label_y, feature_y, _, _ = data
    batch_size = label_x.shape[0]
    encoder_hidden = encoder.initHidden(batch_size)
    loss = 0.
    outputs = []
    label_x, feature_x, label_y, feature_y, _, _ = data

    inp = torch.cat(
        [label_x[:, :-1].reshape(label_x.shape[0], label_x.shape[1] - 1, 1), feature_x[:, 1:, :]],
        dim=2)

    for time_step in range(window - 1):
        output, encoder_hidden = encoder(inp[:, time_step:time_step + 1, :], encoder_hidden)

    decoder_hidden = encoder_hidden

    for idx in range(forecast_length):
        if idx == 0:
            inp = torch.cat([label_x[:, -1].reshape([-1, 1, 1]), feature_y[:, idx:idx + 1, :]],
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

        output, decoder_hidden = decoder(inp, decoder_hidden)
        outputs.append(output)

    outputs = torch.stack(outputs, 1)
    loss = loss_fn(outputs, label_y)
    avg_acc = accuracy(outputs, label_y)
    return loss, outputs, avg_acc


def evaluate(val_loader, model, loss_fn, window, forecast_length):
    losses = []
    accs = []
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            loss, output, acc = _forward(data, model, loss_fn, window, forecast_length,
                                         training=False)
            losses.append(loss)
            accs.append(acc)
    return np.mean(losses), np.mean(accs)


def train(raw, flags):
    data_processor = DataProcessor(flags.forecast_length, flags.batch_size, flags.window)
    train_loader, val_loader = data_processor.get_train_test_data(raw,
                                                                  flags.validation_ratio)

    encoder = EncoderRNN(data_processor.num_features + 1, flags.num_units)
    decoder = DecoderRNN(data_processor.num_features + 1, flags.num_units,
                         output_size=flags.output_dim, batch_first=True)

    def init_weights(m):
        for name, param in m.named_parameters():
            torch.nn.init.uniform_(param.data, -0.1, 0.1)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = SMAPE()
    model_params = list(encoder.parameters()) + list(decoder.parameters())

    opt = torch.optim.Adam(model_params, lr=flags.learning_rate)

    teacher_ratio = flags.teacher_ratio
    loss_history = []

    # model, opt, start_epoch = load_checkpoint(flags.checkpoint_path, model, opt)
    # if start_epoch >= flags.num_epochs:
    #     print('start_epoch is larger than num_epochs!')
    start_epoch = 0
    epoch = start_epoch
    # TODO: add early stop
    for epoch in range(start_epoch, flags.num_epochs):
        for step, data in enumerate(train_loader):
            avg_loss, _, acc = _forward(data, [encoder, decoder], loss_fn, flags.window,
                                        flags.forecast_length, True, teacher_ratio=teacher_ratio)
            loss_history.append(avg_loss)
            opt.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 5.0)
            opt.step()
            teacher_ratio *= flags.teacher_ratio_decay
        val_loss, val_acc = evaluate(val_loader, [encoder, decoder], loss_fn, flags.window,
                                     flags.forecast_length)
        print('Epoch: %d' % epoch)
        print("Training Loss:%.3f" % avg_loss)
        print('Training Avg Accuracy:%.3f' % acc)
        print("Validation Loss:%.3f" % val_loss)
        print("Validation Accuracy:%.3f" % val_acc)
        print('Teacher_ratio: %.3f' % teacher_ratio)
        # print('Model training completed and save at %s' % flags.checkpoint_path)  # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}  # if not os.path.exists(flags.checkpoint_path):  #     os.mkdir(flags.checkpoint_path)  # torch.save(state, flags.checkpoint_path + '/model.pt')  # data_processor.save(flags.checkpoint_path)  # return model, loss_history
        print('Gradients:%.3f' % torch.mean(
            (torch.stack([torch.mean(torch.abs(p.grad)) for p in model_params], 0))))
        print()


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
    loss_fn = SMAPE()
    model, _, _ = load_checkpoint(flags.checkpoint_path, model, None)
    model.eval()
    # predict
    results = {}
    loader, ts_types = data_processor.get_forecast_data(raw)
    with torch.no_grad():
        for type, data in zip(ts_types, loader):
            scale = data[4]
            _, outputs = _forward(data, model, loss_fn, flags.window, flags.forecast_length, 1.)
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
