import argparse
import cv2
import os
from matplotlib import axes
import numpy as np
import pandas as pd
from numpy.core.defchararray import title

import transform

COLORS = 256
CROP = 80


def hist_count(index, img):
    img = img[index]
    hist = np.zeros(COLORS)
    for c in range(COLORS):
        count = np.sum(img == c)
        hist[c] = count
    return hist


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, help='Path to an image')

args = parser.parse_args()

images = []
labels = []

for file_name in os.listdir(args.data_dir + '/images'):
    image = cv2.imread(args.data_dir + '/images/' + file_name, cv2.IMREAD_GRAYSCALE)[
        0:-80, :].astype('int32')
    images.append(image)

for file_name in os.listdir(args.data_dir + '/labels'):
    label = cv2.imread(args.data_dir + '/labels/' + file_name,
                       cv2.IMREAD_COLOR)[0:-80, :].astype('int32')
    labels.append(label)

pores_hist = np.zeros(COLORS)
organic_matter_hist = np.zeros(COLORS)
clay_hist = np.zeros(COLORS)
mineral_grain_hist = np.zeros(COLORS)

for i in range(len(images)):
    label = labels[i]
    image = images[i]

    pores_index = np.all(label == transform.GREEN, axis=2)
    organic_matter_index = np.all(label == transform.RED, axis=2)
    clay_index = np.all(label == transform.BLACK, axis=2)
    mineral_grain_index = (pores_index == False) & (
        organic_matter_index == False) & (clay_index == False)

    pores_hist += hist_count(pores_index, image)
    organic_matter_hist += hist_count(organic_matter_index, image)
    clay_hist += hist_count(clay_index, image)
    mineral_grain_hist += hist_count(mineral_grain_index, image)

pores_hist /= len(images) * images[0].size
organic_matter_hist /= len(images) * images[0].size
clay_hist /= len(images) * images[0].size
mineral_grain_hist /= len(images) * images[0].size

colors = ['red', 'green', 'black', 'yellow']
map_colors = ['orgainic matter', 'pores', 'clay', 'mineral grain']

y = np.zeros((COLORS, 4))
y[:, 0] = organic_matter_hist
y[:, 1] = pores_hist
y[:, 2] = clay_hist
y[:, 3] = mineral_grain_hist

df = pd.DataFrame(y, columns=map_colors)
ax = df.plot(color=colors, kind="bar", title='Gray Scale Analysis',
             ylabel='Frequency', xlabel='Color', xticks=[0, 50, 100, 150, 200, 250])
fig = ax.get_figure()
fig.savefig('figure2.png')
