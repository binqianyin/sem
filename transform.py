import torch

# RGB colors
GREEN = [0, 255, 0]
RED = [0, 0, 255]
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
YELLOW = [0, 255, 255]
COLOR_DIST = 0


def transpose(images, hwc=True):
    '''
       Transpose from hwc to chw or chw to hwc

       :images: a list of torch tensors
       :hwc: boolean, True if hwc, False if chw

       :return: a list of torch tensors
    '''
    new_images = []
    for image in images:
        if hwc:
            # hwc to chw
            new_image = image.permute(2, 0, 1)
        else:
            # chw to hwc
            new_image = image.permute(1, 2, 0)
        new_images.append(new_image)
    return new_images


def normalize(images):
    '''
       Normalize images using 255.0 for each channel

       :images: a list of torch tensors

       :return: a list of torch tensors
    '''
    new_images = []
    for image in images:
        new_image = torch.div(image, 255.0)
        new_images.append(new_image)
    return new_images


def denormalize(images):
    '''
       Denormalize images using 255.0 for each channel

       :images: a list of torch tensors

       :return: a list of torch tensors
    '''
    new_images = []
    for image in images:
        new_image = torch.mul(image, 255.0)
        new_images.append(new_image)
    return new_images


def prepare_binary_labels(labels):
    '''
       Return three labels arrays

       :labels: a list of torch tensors

       :return: multiple lists of torch tensors representing four categories
    '''
    white_labels = []
    red_labels = []
    green_labels = []
    black_labels = []
    white_color = torch.Tensor(WHITE)
    green_color = torch.Tensor(GREEN)
    red_color = torch.Tensor(RED)
    black_color = torch.Tensor(BLACK)
    for label in labels:
        white_index = torch.sum(label == white_color, axis=2) == 3
        green_index = torch.sum(label == green_color, axis=2) == 3
        red_index = torch.sum(label == red_color, axis=2) == 3
        black_index = torch.sum(label == black_color, axis=2) == 3
        white_labels.append(white_index.float().unsqueeze(-1))
        red_labels.append(red_index.float().unsqueeze(-1))
        green_labels.append(green_index.float().unsqueeze(-1))
        black_labels.append(black_index.float().unsqueeze(-1))
    return white_labels, red_labels, green_labels, black_labels


def prepare_multi_labels(labels):
    '''
       Return a single label arrays

       :labels: a list of torch tensors

       :return: a list of torch tensors
    '''
    normalized_labels = []
    green_color = torch.Tensor(GREEN)
    red_color = torch.Tensor(RED)
    black_color = torch.Tensor(BLACK)
    for label in labels:
        red_index = torch.sum(label == red_color, axis=2) == 3
        green_index = torch.sum(label == green_color, axis=2) == 3
        black_index = torch.sum(label == black_color, axis=2) == 3
        white_index = (red_index == 0) & (
            green_index == 0) & (black_index == 0)
        normalized_label = torch.LongTensor(red_index.shape)
        normalized_label[white_index] = 0
        normalized_label[red_index] = 1
        normalized_label[green_index] = 2
        normalized_label[black_index] = 3
        normalized_labels.append(normalized_label)
    return normalized_labels


def convert_labels(probs, labels):
    '''
       Convert labels to images

       :probs: output probability at each pixel
       :labels: a list of torch tensors

       :return: a list of torch tensors
    '''
    images = []
    red_color = torch.Tensor(RED)
    green_color = torch.Tensor(GREEN)
    black_color = torch.Tensor(BLACK)
    white_color = torch.Tensor(WHITE)
    for i, label in enumerate(labels):
        prob = probs[i]
        image = torch.Tensor(label.shape[0], label.shape[1], 3)
        red_index = (prob > 0.5) & (label == 1)
        green_index = (prob > 0.5) & (label == 2)
        black_index = (prob > 0.5) & (label == 3)
        white_index = (red_index == 0) & (
            green_index == 0) & (black_index == 0)
        image[red_index] = red_color
        image[green_index] = green_color
        image[black_index] = black_color
        image[white_index] = white_color
        images.append(image)
    return images


def convert_multi_labels(multi_labels, label_orders=[[0, 1, 2, 3]], classes=4):
    '''
       Convert labels to images

       :labels: a list of torch tensors
       :label_orders: a list of lists of integers representing the order of the labels
       :classes: number of classes

       :return: a list of torch tensors
    '''
    images = []
    red_color = torch.Tensor(RED)
    green_color = torch.Tensor(GREEN)
    black_color = torch.Tensor(BLACK)
    white_color = torch.Tensor(WHITE)
    yellow_color = torch.Tensor(YELLOW)
    for i, multi_label in enumerate(multi_labels):
        image = torch.Tensor(multi_label.shape[0], multi_label.shape[1], 3)
        if len(label_orders) == 1:
            white_index = multi_label == label_orders[0][0]
            red_index = multi_label == label_orders[0][1]
            green_index = multi_label == label_orders[0][2]
            black_index = multi_label == label_orders[0][3]
        else:
            white_index = multi_label == label_orders[i][0]
            red_index = multi_label == label_orders[i][1]
            green_index = multi_label == label_orders[i][2]
            black_index = multi_label == label_orders[i][3]
        image[red_index] = red_color
        image[green_index] = green_color
        image[black_index] = black_color
        image[white_index] = white_color
        image[~(red_index | green_index | black_index |
                white_index)] = yellow_color
        images.append(image)
    return images


def regularize(images):
    '''
      Generate images contain only red, green, black, and white colors

      :images: a list of torch tensors

      :return: a list of torch tensors representing only red, green, black, and white colors
    '''
    new_images = []
    green_color = torch.Tensor(GREEN)
    red_color = torch.Tensor(RED)
    black_color = torch.Tensor(BLACK)
    white_color = torch.Tensor(WHITE)
    for image in images:
        new_image = torch.Tensor(image.shape)
        green_dist = torch.sqrt(
            torch.sum(torch.pow(image - green_color, 2), dim=2))
        red_dist = torch.sqrt(
            torch.sum(torch.pow(image - red_color, 2), dim=2))
        black_dist = torch.sqrt(
            torch.sum(torch.pow(image - black_color, 2), dim=2))

        green_index = green_dist <= COLOR_DIST
        red_index = red_dist <= COLOR_DIST
        black_index = black_dist <= COLOR_DIST
        white_index = (green_index == 0) & (
            red_index == 0) & (black_index == 0)

        new_image[green_index] = green_color
        new_image[red_index] = red_color
        new_image[black_index] = black_color
        new_image[white_index] = white_color

        new_images.append(new_image)
    return new_images


def crop(images, dims):
    '''
      Crop images on four dimensions according to dim

      :images: a list of torch tensors
      :dims: dimensions of the cropped area

      :return: a list of torch tensors
    '''
    new_images = []
    up = dims[0]
    right = dims[1]
    down = dims[2]
    left = dims[3]
    for image in images:
        rows = image.shape[0]
        cols = image.shape[1]
        new_image = image[up:rows - down, left:cols - right]
        new_images.append(new_image)
    return new_images


def augment(images, labels, dims, stride=1, flip=False, rotate=False):
    '''
      Slice images and labels and generate sub images with dims.
      Move stride to generate the next image.

      :images: a list of torch tensors
      :dims: dimensions of the generated images
      :labels: a list of torch tensors
      :stride: stride between new images on the raw image
      :flip: horizontal and vertical flip of new images
      :rotate: rotate 90, 180, and 270 degress

      :return: a list of torch tensors representing subimages, and a list of torch tensors representing labels
    '''
    new_images = []
    new_labels = []
    for image_idx in range(len(images)):
        image = images[image_idx]
        label = labels[image_idx]
        sub_image_rows = (image.shape[0] - dims[0]) // stride + 1
        sub_image_cols = (image.shape[1] - dims[1]) // stride + 1
        for i in range(sub_image_rows):
            stride_begin_row = i * stride
            stride_end_row = stride_begin_row + dims[0]
            for j in range(sub_image_cols):
                stride_begin_col = j * stride
                stride_end_col = stride_begin_col + dims[1]
                img = image[stride_begin_row:stride_end_row,
                            stride_begin_col:stride_end_col]
                new_images.append(img)
                lab = label[stride_begin_row:stride_end_row,
                            stride_begin_col:stride_end_col]
                new_labels.append(lab)
    return new_images, new_labels


def cutout(images, labels, threshold=1.0):
    '''
      Throws images with one category greater than threshold

      :images: a list of torch tensors
      :labels: a list of torch tensors
      :threshold: ratio of the most color

      :return: lists of torch tensors representing images with one category greater than threshold
    '''
    new_images = []
    new_labels = []
    green_color = torch.Tensor(GREEN)
    red_color = torch.Tensor(RED)
    black_color = torch.Tensor(BLACK)
    white_color = torch.Tensor(WHITE)
    for image_idx in range(len(images)):
        image = images[image_idx]
        label = labels[image_idx]
        green = torch.sum(torch.sum(label == green_color, axis=-1) == 3)
        red = torch.sum(torch.sum(label == red_color, axis=-1) == 3)
        black = torch.sum(torch.sum(label == black_color, axis=-1) == 3)
        white = torch.sum(torch.sum(label == white_color, axis=-1) == 3)
        size = label.shape[0] * label.shape[1]
        if green < size * threshold and red < size * threshold and \
                black < size * threshold and white < size * threshold:
            new_images.append(image)
            new_labels.append(label)
    return new_images, new_labels


def flatten(images):
    '''
      transform N images with shape [H, W, C] to a [N * H * W, C] tensor

      :images: a list of torch tensors

      :return: a single torch tensor
    '''
    for i in range(len(images)):
        if len(images[i].shape) < 3:
            images[i] = images[i].unsqueeze(-1)
    ret = torch.zeros((len(images) * images[0].shape[0] * images[0].shape[1],
                       images[0].shape[2]))
    for i in range(len(images)):
        # [H, W, C] -> [H * W, C]
        image = images[i].view(images[i].shape[0] * images[i].shape[1], -1)
        ret[i * images[i].shape[0] * images[i].shape[1]            :(i + 1) * images[i].shape[0] * images[i].shape[1], :] = image[:]
    return ret


def unflatten(images):
    '''
      transform a [N * H * W, C] tensor into N images with shape [H, W, C]

      :return: a single torch tensor

      :images: a list of torch tensors
    '''
    ret = []
    for i in range(len(images)):
        # [H * W, C] -> [H, W, C]
        image = images[i].view(images[i].shape[0] // images[i].shape[0],
                               images[i].shape[0], -1)
        ret.append(image[:])
    return ret
