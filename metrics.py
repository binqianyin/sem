from ctypes import Union
import os
from pickle import LIST
import torch
import math
import shutil
import json


def accuracy(predicts, truths):
    '''
      Calculate the matches between predicts and truths

      :predicts: A list of torch Tensors
      :truths: A list of torch Tensors
    '''
    matches = torch.sum(truths == (predicts > 0.5))
    return matches / len(truths)


def binary_iou(predict, truth):
    '''
       Calculate the IOU of predict and truth

       :predict: A torch Tensor
       :truth: A torch Tensor
    '''
    intersection = predict & truth
    union = predict | truth
    intersection_sum = torch.sum(intersection)
    union_sum = torch.sum(union)
    if union_sum.item() == 0:
        return torch.tensor([1.0], device=predict.device)
    else:
        return intersection_sum / union_sum


def average_iou(predicts, truths, labels):
    '''
       Calculate the averaged IOU of multiple classes

      :predicts: A list of torch Tensors
      :truths: A list of torch Tensors
      :labels: A list of category values
    '''
    ious = []
    for i, predict in enumerate(predicts):
        truth = truths[i]
        iou = 0.0
        for label in labels:
            predict_index = torch.all(predict == label, axis=-1)
            truth_index = torch.all(truth == label, axis=-1)
            iou += binary_iou(predict_index, truth_index).item()
        ious.append(iou / len(labels))
    return ious


def average_iou_tensor(predicts, truths, labels):
    '''
       Calculate the averaged IOU of multiple classes

      :predicts: A list of torch Tensors
      :truths: A list of torch Tensors
      :labels: A list of category values
    '''
    iou = 0.0
    ious = []
    for label in labels:
        predict_index = predicts == label
        truth_index = truths == label
        biou = binary_iou(predict_index, truth_index).item()
        ious.append(biou)
        iou += biou
    return iou / len(labels), ious


def weight_bce_loss(output, labels, bce):
    total_colors = labels.numel()
    true_colors = torch.sum(labels)
    # 1. WCE simple
    weight = (labels == 1.0) * (total_colors / true_colors)
    weight[labels == 0.0] = 1.0
    # 2. WCE adaptive
    #weight = torch.ones_like(labels)
    # weight_index_pos = (labels == 1.0).type(
    #    torch.BoolTensor).to(output.device) & (output < 0.5)
    # weight_index_neg = (labels == 0.0).type(
    #    torch.BoolTensor).to(output.device) & (output > 0.5)
    #weight[weight_index_pos] *= (total_colors / true_colors)
    #weight[weight_index_neg] *= (total_colors / true_colors)
    # 3. Adpative
    #weight = torch.exp(-math.log(100) * torch.pow(output, 2))
    loss = bce(output, labels)
    loss = loss * weight
    loss = torch.mean(loss)
    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, device='cuda'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction='none')
        at = (target == 1.0) * self.alpha
        at[target == 0.0] = (1 - self.alpha)
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        F_loss = at * (1 - pt)**self.gamma * bce_loss
        return F_loss.mean()


class FocalLossMulti(torch.nn.Module):
    # weight parameter will act as the alpha parameter to balance class weights
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLossMulti, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = torch.nn.functional.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        F_loss = ((1 - pt) ** self.gamma * ce_loss)
        return F_loss.mean()


class FocalDiceLossMulti(torch.nn.Module):
    # weight parameter will act as the alpha parameter to balance class weights
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalDiceLossMulti, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target, smooth=1.0):
        input = torch.softmax(input, dim=1).permute(0, 2, 3, 1)
        target = torch.nn.functional.one_hot(
            target, num_classes=input.shape[-1])

        if self.weight is not None:
            intersection = (self.weight * input * target).sum(dim=-1)
        else:
            intersection = (input * target).sum(dim=-1)
        dice = 1 - (2.*intersection + smooth) / \
            (input.sum(dim=-1) + target.sum(dim=-1) + smooth)

        pt = torch.exp(-dice)
        F_loss = ((1 - pt) ** self.gamma * dice)
        return F_loss.mean()


def color_weight(multi_labels):
    # white, red, green, black
    num_white = 0
    num_red = 0
    num_green = 0
    num_black = 0
    for multi_label in multi_labels:
        num_white += torch.sum(multi_label == 0).item()
        num_red += torch.sum(multi_label == 1).item()
        num_green += torch.sum(multi_label == 2).item()
        num_black += torch.sum(multi_label == 3).item()
    sum_colors = num_white + num_red + num_green + num_black
    weight_white = (sum_colors - num_white) / float(sum_colors)
    weight_red = (sum_colors - num_red) / float(sum_colors)
    weight_green = (sum_colors - num_green) / float(sum_colors)
    weight_black = (sum_colors - num_black) / float(sum_colors)
    return torch.tensor([weight_white, weight_red, weight_green, weight_black])


class ExpDict:
    def __init__(self, name: str = None) -> None:
        self._name = name
        self._data = dict()

    def add_metric(self, exp_name: str, entry_name: str, metric) -> None:
        if exp_name not in self._data:
            self._data[exp_name] = dict()

        if entry_name not in self._data[exp_name]:
            self._data[exp_name][entry_name] = []

        self._data[exp_name][entry_name].append(metric)

    @property
    def data(self):
        return self._data

    def read(self, path):
        with open(path + '/' + self._name, 'r') as f:
            self._data = json.load(f)
        return self._data

    def dump(self, path):
        with open(path + '/' + self._name, 'w') as f:
            json.dump(self._data, f)

# PyTorch


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.nn.functional.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy(
            inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, device='cuda'):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, smooth=1.0):
        input = torch.sigmoid(input)
        intersection = (input * target).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (input.sum() + target.sum() + smooth)
        # bce_loss = torch.nn.functional.binary_cross_entropy(
        #    input, target, reduction='none') + dice_loss
        at = (target == 1.0) * self.alpha
        at[target == 0.0] = (1 - self.alpha)
        pt = torch.exp(-dice_loss)  # prevents nans when probability 0
        F_loss = at * (1 - pt)**self.gamma * dice_loss
        return F_loss.mean()


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=2.0):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky
