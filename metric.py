import json

import torch


def accuracy(predicts, truths):
    '''
      Calculate the matches between predicts and truths

      :param predicts: A list of torch Tensors
      :param truths: A list of torch Tensors

      :return: A list of accuracy values
    '''
    matches = torch.sum(truths == (predicts > 0.5))
    return matches / len(truths)


def binary_iou(predict, truth):
    '''
       Calculate the IOU of predict and truth

       :param predict: A torch Tensor
       :param truth: A torch Tensor

       :return: A torch Tensor
    '''
    intersection = predict & truth
    union = predict | truth
    intersection_sum = torch.sum(intersection).item()
    union_sum = torch.sum(union).item()
    if union_sum == 0:
        return 1.0
    else:
        return intersection_sum / union_sum


def average_iou(predicts, truths, labels):
    '''
      Calculate the averaged IOU of multiple classes

      :param predicts: A list of torch Tensors
      :param truths: A list of torch Tensors
      :param labels: A list of category values

      :return: A list of accuracy values
    '''
    ious = []
    for i, predict in enumerate(predicts):
        truth = truths[i]
        iou = 0.0
        for label in labels:
            predict_index = torch.all(predict == label, axis=-1)
            truth_index = torch.all(truth == label, axis=-1)
            iou += binary_iou(predict_index, truth_index)
        ious.append(iou / len(labels))
    return ious


def average_iou_tensor(predicts, truths, labels):
    '''
      Calculate the averaged IOU of multiple classes

      :param predicts: A torch tensor whose first dimension is the batch size
      :param truths: A torch tensor whose first dimension is the batch size
      :param labels: A list of category values

      :return: A list of accuracy values
    '''
    iou = 0.0
    ious = []
    for label in labels:
        predict_index = predicts == label
        truth_index = truths == label
        biou = binary_iou(predict_index, truth_index)
        ious.append(biou)
        iou += biou
    return iou / len(labels), ious


def weight_bce_loss(output, labels, bce):
    '''
        Calculate the weighted binary cross entropy loss

        :param output: A torch tensor
        :param labels: A torch tensor
        :param bce: A torch binary cross entropy loss function

        :return: A torch tensor
    '''
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
    '''
        Implement the focal loss function for binary classification

        Paper reference: https://arxiv.org/abs/1708.02002
    '''

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
    '''
        Implement the focal loss function for classification of multiple classes

        Paper reference: https://arxiv.org/abs/1708.02002
    '''
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


class FocalDiceLoss(torch.nn.Module):
    '''
        Combine focal loss and dice loss to predict binary classification

        Dice loss is used to balance the class weights
        Focal loss is used to focus on difficult classes

        Focal loss paper reference: https://arxiv.org/abs/1708.02002
        Dice loss paper reference: https://arxiv.org/abs/1511.00561
    '''

    def __init__(self, alpha=0.25, gamma=2.0, device='cuda'):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target, smooth=1.0):
        input = torch.sigmoid(input)
        intersection = (input * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / \
            (input.sum() + target.sum() + smooth)
        # bce_loss = torch.nn.functional.binary_cross_entropy(
        #    input, target, reduction='none') + dice_loss
        at = (target == 1.0) * self.alpha
        at[target == 0.0] = (1 - self.alpha)
        pt = torch.exp(-dice_loss)  # prevents nans when probability 0
        F_loss = at * (1 - pt)**self.gamma * dice_loss
        return F_loss.mean()


class FocalDiceLossMulti(torch.nn.Module):
    '''
        Combine focal loss and dice loss to predict multiple classes

        Dice loss is used to balance the class weights
        Focal loss is used to focus on difficult classes

        Focal loss paper reference: https://arxiv.org/abs/1708.02002
        Dice loss paper reference: https://arxiv.org/abs/1511.00561
    '''
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
        dice = 1 - (2. * intersection + smooth) / \
            (input.sum(dim=-1) + target.sum(dim=-1) + smooth)

        pt = torch.exp(-dice)
        F_loss = ((1 - pt) ** self.gamma * dice)
        return F_loss.mean()


def color_weight(multi_labels):
    '''
        Calculate the weight of each class

        :param multi_labels: A list of torch tensor representing color labels

        :return: A torch tensor
    '''
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
    '''
        A dictionary that can be used to store the experiment results
    '''

    def __init__(self, name: str = None) -> None:
        self._name = name
        self._data = dict()

    def add_metric(self, exp_name: str, entry_name: str, metric) -> None:
        '''
            Add a metric to the dictionary

            :param exp_name: The name of the experiment
            :param entry_name: The name of the entry
            :param metric: The metric to be added

            :return: None
        '''
        if exp_name not in self._data:
            self._data[exp_name] = dict()

        if entry_name not in self._data[exp_name]:
            self._data[exp_name][entry_name] = []

        self._data[exp_name][entry_name].append(metric)

    @property
    def data(self):
        return self._data

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def read(self, path):
        '''
            Read the data from a file

            :param path: The path to the file

            :return: ExpDict object
        '''
        with open(path + '/' + self._name, 'r') as f:
            self._data = json.load(f)
        return self

    def dump(self, path):
        '''
            Dump the data to a file

            :param path: The path to the file
        '''
        with open(path + '/' + self._name, 'w') as f:
            json.dump(self._data, f)
