import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import metric


def plot(ax, name, curves: dict, step=5):
    y = np.transpose(np.array(list(curves.values())))
    x = np.arange(y.shape[0])
    x = x[0:-1:step]
    y = y[0:-1:step, :]
    ax.plot(x, y[:, 0], 'y+-')
    ax.plot(x, y[:, 1], 'm*-')
    ax.set_title(name)
    ax.set_xlabel('iterations')
    ax.set_ylabel('iou')


parser = argparse.ArgumentParser()
parser.add_argument('--exp-dir', type=str, help='Path to experiment files')
parser.add_argument('--step', type=int, default=1)
args = parser.parse_args()

train_exp_dict = metric.ExpDict('backbone.json').read(args.exp_dir)
train_multi_exp_dict = metric.ExpDict('multi.json').read(args.exp_dir)

colors = ['white', 'red', 'green', 'black']
color_map = ['Mineral Grain', 'Organic Matter', 'Pores', 'Clay']
locs = [(0, 0), (0, 1), (1, 0), (1, 1)]

_, axs = plt.subplots(2, 2, figsize=(12, 12))
for i, color in enumerate(colors):
    curves = dict()
    curves['UNet'] = np.array(
        train_multi_exp_dict['multi']['train_ious'])[:, i]
    curves['Paw-Net'] = np.array(
        train_exp_dict[color]['train_iou'])
    plot(axs[locs[i]], color_map[i], curves, args.step)
lg = plt.legend(list(curves.keys()), bbox_to_anchor=(
    1.04, 1.2), loc="upper left", prop={'size': 16})
plt.savefig('figure7.png', bbox_extra_artists=(lg,), bbox_inches='tight')
plt.close()
