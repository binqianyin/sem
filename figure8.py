import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import metric


def plot(title, name, loss_curves: dict, iou_curves: dict):
    train_y = np.transpose(np.array(list(loss_curves.values())))
    train_x = np.arange(train_y.shape[0])
    fig, ax1 = plt.subplots()
    ax1.plot(train_x, train_y[:, 0], 'bo-')
    ax1.plot(train_x, train_y[:, 1], 'gv-')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss')

    train_y = np.transpose(np.array(list(iou_curves.values())))
    train_x = np.arange(train_y.shape[0])
    ax2 = ax1.twinx()
    ax2.plot(train_x, train_y[:, 0], 'y+-')
    ax2.plot(train_x, train_y[:, 1], 'm*-')
    ax2.plot(np.nan, 'bo-')
    ax2.plot(np.nan, 'gv-')
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('iou')
    ax2.legend(list(iou_curves.keys()) +
               list(loss_curves.keys()), loc=(0.7, 0.1))
    plt.title(title)
    plt.savefig(name)


parser = argparse.ArgumentParser()
parser.add_argument('--exp-dir', type=str, help='Path to experiment files')
args = parser.parse_args()

train_multi_exp_dict = metric.ExpDict('multi.json').read(args.exp_dir)
train_syn_exp_dict = metric.ExpDict(
    'synthesizer.json').read(args.exp_dir)

loss_curves = dict()
iou_curves = dict()

loss_curves['UNet loss'] = np.array(
    train_multi_exp_dict['multi']['train_loss'])
loss_curves['Paw-Net loss'] = np.array(
    train_syn_exp_dict['synthesizer']['train_loss'])
iou_curves['UNet iou'] = np.array(
    train_multi_exp_dict['multi']['train_iou'])
iou_curves['Paw-Net iou'] = np.array(
    train_syn_exp_dict['synthesizer']['train_iou'])
plot('training loss and iou curves', 'figure8-1.png',
     loss_curves, iou_curves=iou_curves)

loss_curves = dict()
iou_curves = dict()

loss_curves['UNet loss'] = np.array(
    train_multi_exp_dict['multi']['validate_loss'])
loss_curves['Paw-Net loss'] = np.array(
    train_syn_exp_dict['synthesizer']['validate_loss'])
iou_curves['UNet iou'] = np.array(
    train_multi_exp_dict['multi']['validate_iou'])
iou_curves['Paw-Net iou'] = np.array(
    train_syn_exp_dict['synthesizer']['validate_iou'])
plot('validation loss and iou curves', 'figure8-2.png',
     loss_curves, iou_curves=iou_curves)
