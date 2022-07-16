import argparse
import time

import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import images_io
import metric
import transform

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', required=True,
                    help='Directory that contains all training labels and images')
parser.add_argument('--test-dir', required=True,
                    help='Directory that contains all test labels and images')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

images, _ = images_io.read_images(args.train_dir + '/images')
labels, _ = images_io.read_images(args.train_dir + '/labels')
labels = transform.prepare_multi_labels(labels)

# train
start = time.time()
X = transform.flatten(images).numpy()
Y = transform.flatten(labels).squeeze(-1).numpy()
clf = LinearDiscriminantAnalysis()
clf.fit(X, Y)
preds = torch.from_numpy(clf.predict(X)).to(args.device).to(args.device)
iou, ious = metric.average_iou_tensor(
    preds, torch.from_numpy(Y).to(args.device), [0, 1, 2, 3])
end = time.time()
print('Training Time:', end - start)
print('Training IOU:', iou)

# test
start = time.time()
test_images, _ = images_io.read_images(args.test_dir + '/images')
test_labels, _ = images_io.read_images(args.test_dir + '/labels')
test_labels = transform.prepare_multi_labels(test_labels)
test_X = transform.flatten(test_images).numpy()
preds = torch.from_numpy(clf.predict(test_X)).to(args.device)
test_Y = transform.flatten(test_labels).squeeze(-1).to(args.device)
iou, ious = metric.average_iou_tensor(preds, test_Y, [0, 1, 2, 3])
end = time.time()
print('Test Time:', end - start)
print('Test IOU:', iou)
