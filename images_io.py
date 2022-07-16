import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_images(path):
    '''
        Read the images from a folder

        :param path: The path to the folder

        :return: A list of images and a list of their names
    '''
    images = []
    file_names = []
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            image = cv2.imread(path + '/' + file_name)
            image = torch.from_numpy(np.array(image.astype('float32')))
            images.append(image)
            file_names.append(file_name)
    return images, file_names


def write_images(path, images):
    '''
        Write the images to a folder

        :param path: The path to the folder
        :param images: The images to be written
    '''
    # if delete is True and os.path.isdir(path):
    #    shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    for i, image in enumerate(images):
        cv2.imwrite(path + '/' + str(i) + '.tif',
                    image.numpy().astype('uint8'))


class ImageDataset(Dataset):
    '''
        Get the images and labels
    '''
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        if self._labels is not None:
            label = self._labels[idx]
            return image, label
        else:
            return image
