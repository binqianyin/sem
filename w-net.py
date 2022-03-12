import argparse
import torch
import torch.nn as nn
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

import images_io
import transform
import metrics


class Config():
    def __init__(self):
        self.channels = 3
        self.k = 4  # Number of classes

        self.useInstanceNorm = True  # Instance Normalization
        self.useBatchNorm = False  # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.2

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256]
        self.decoderLayerSizes = [512, 256]

        self.variationalTranslation = 0  # Pixels, 0 for off. 1 works fine

        self.saveModel = True


config = Config()


def soft_n_cut_loss(inputs, segmentations):
    # We don't do n_cut_loss batch wise -- split it up and do it instance wise
    loss = 0
    for i in range(inputs.shape[0]):
        flatten_image = torch.mean(inputs[i], dim=0)
        flatten_image = flatten_image.reshape(flatten_image.shape[0]**2)
        loss += soft_n_cut_loss_(flatten_image,
                                 segmentations[i], config.k, inputs[0].shape[1], inputs[0].shape[2])
    loss = loss / inputs.shape[0]
    return loss


def soft_n_cut_loss_(flatten_image, prob, k, rows, cols):
    '''
    Inputs:
    prob : (rows*cols*k) tensor
    k : number of classes (integer)
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    rows : number of the rows in the original image
    cols : number of the cols in the original image
    Output :
    soft_n_cut_loss tensor for a single image
    '''

    loss = k
    weights = edge_weights(flatten_image, rows, cols)

    for t in range(k):
        loss = loss - \
            (numerator(prob[t, :, ], weights) /
             denominator(prob[t, :, :], weights))

    return loss


def edge_weights(flatten_image, rows, cols, std_intensity=3, std_position=1, radius=5):
    '''
    Inputs :
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    std_intensity : standard deviation for intensity
    std_position : standard devistion for position
    radius : the length of the around the pixel where the weights
    is non-zero
    rows : rows of the original image (unflattened image)
    cols : cols of the original image (unflattened image)
    Output :
    weights :  2d tf array edge weights in the pixel graph
    Used parameters :
    n : number of pixels
    '''
    ones = torch.ones_like(flatten_image, dtype=torch.float)
    if torch.cuda.is_available():
        ones = ones.cuda()

    A = outer_product(flatten_image, ones)
    A_T = torch.t(A)
    d = torch.div((A - A_T), std_intensity)
    intensity_weight = torch.exp(-1*torch.mul(d, d))

    xx, yy = torch.meshgrid(torch.arange(
        rows, dtype=torch.float), torch.arange(cols, dtype=torch.float))
    xx = xx.reshape(rows*cols)
    yy = yy.reshape(rows*cols)
    if torch.cuda.is_available():
        xx = xx.cuda()
        yy = yy.cuda()
    ones_xx = torch.ones_like(xx, dtype=torch.float)
    ones_yy = torch.ones_like(yy, dtype=torch.float)
    if torch.cuda.is_available():
        ones_yy = ones_yy.cuda()
        ones_xx = ones_xx.cuda()
    A_x = outer_product(xx, ones_xx)
    A_y = outer_product(yy, ones_yy)

    xi_xj = A_x - torch.t(A_x)
    yi_yj = A_y - torch.t(A_y)

    sq_distance_matrix = torch.mul(xi_xj, xi_xj) + torch.mul(yi_yj, yi_yj)

    # Might have to consider casting as float32 instead of creating meshgrid as float32

    dist_weight = torch.exp(-torch.div(sq_distance_matrix, std_position**2))
    weight = torch.mul(intensity_weight, dist_weight)  # Element wise product

    # ele_diff = tf.reshape(ele_diff, (rows, cols))
    # w = ele_diff + distance_matrix
    return weight


def outer_product(v1, v2):
    '''
    Inputs:
    v1 : m*1 tf array
    v2 : m*1 tf array
    Output :
    v1 x v2 : m*m array
    '''
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    v1 = torch.unsqueeze(v1, dim=0)
    v2 = torch.unsqueeze(v2, dim=0)
    return torch.matmul(torch.t(v1), v2)


def numerator(k_class_prob, weights):
    '''
    Inputs :
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights n*n tensor
    '''
    k_class_prob = k_class_prob.reshape(-1)
    a = torch.mul(weights, outer_product(k_class_prob, k_class_prob))
    return torch.sum(a)


def denominator(k_class_prob, weights):
    '''
    Inputs:
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights	n*n tensor
    '''
    k_class_prob = k_class_prob.view(-1)
    return torch.sum(
        torch.mul(
            weights,
            outer_product(
                k_class_prob,
                torch.ones_like(k_class_prob)
            )
        )
    )


''' Each module consists of two 3 x 3 conv layers, each followed by a ReLU
non-linearity and batch normalization.

In the expansive path, modules are connected via transposed 2D convolution
layers.

The input of each module in the contracting path is also bypassed to the
output of its corresponding module in the expansive path

we double the number of feature channels at each downsampling step
We halve the number of feature channels at each upsampling step

'''

# NOTE: batch norm is up for debate
# We want batch norm if possible, but the batch size is too low to benefit
# So instead we do instancenorm

# Note: Normalization should go before ReLU

# Padding=1 because (3x3) conv leaves of 2pixels in each dimension, 1 on each side
# Do we want non-linearity between pointwise and depthwise (separable) conv?
# Do we want non-linearity after upconv?


class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvModule, self).__init__()

        layers = [
            # Pointwise (1x1) through all channels
            nn.Conv2d(input_dim, output_dim, 1),
            # Depthwise (3x3) through each channel
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.Dropout)]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class BaseNet(nn.Module):  # 1 U-net
    def __init__(self, input_channels=3,
                 encoder=[64, 128, 256, 512], decoder=[1024, 512, 256], output_channels=config.k):
        super(BaseNet, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(2, 2)
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder])

        decoder_out_sizes = [int(x/2) for x in decoder]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder])  # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(
            128, 128, 2, stride=2)

        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),

            nn.Conv2d(64, output_channels, 1),  # No padding on pointwise
            nn.ReLU(),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(
                layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(input_channels=config.channels, encoder=config.encoderLayerSizes,
                                 decoder=config.decoderLayerSizes, output_channels=config.k)
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(input_channels=config.k, encoder=config.encoderLayerSizes,
                                 decoder=config.decoderLayerSizes, output_channels=config.channels)
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations

    def forward_decoder(self, segmentations):
        x18 = self.U_decoder(segmentations)
        reconstructions = self.sigmoid(x18)
        return reconstructions

    def forward(self, x):  # x is (3 channels 224x224)
        segmentations = self.forward_encoder(x)
        x_prime = self.forward_decoder(segmentations)
        return segmentations, x_prime


def reconstruction_loss(x, x_prime):
    binary_cross_entropy = F.binary_cross_entropy(
        x_prime, x, reduction='mean')
    return binary_cross_entropy


def train(images, batch_size, epochs, path):
    device = torch.device('cuda')

    valid_size = len(images) // 5
    train_size = int(len(images) - valid_size)

    dataset = images_io.ImageDataset(images, None)
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    autoencoder = WNet()
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters())

    ###################################
    #          Training Loop          #
    ###################################

    for epoch in range(epochs):
        autoencoder.train()
        with tqdm(train_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                inputs = data[0].to(device)

                segmentations, reconstructions = autoencoder(inputs)

                l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)
                l_reconstruction = reconstruction_loss(inputs, reconstructions)

                loss = (l_reconstruction + l_soft_n_cut)
                # We only need to do retain graph =true if we're backpropping from multiple heads
                optimizer.zero_grad()  # Don't change gradient on validation
                loss.backward()
                optimizer.step()
                tepoch.set_description(
                    'Train Unsupervised Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item())

        autoencoder.eval()
        with tqdm(valid_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                inputs = data[0].to(device)

                segmentations, reconstructions = autoencoder(inputs)

                l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)
                l_reconstruction = reconstruction_loss(inputs, reconstructions)

                loss = (l_reconstruction + l_soft_n_cut)
                # We only need to do retain graph =true if we're backpropping from multiple heads
                optimizer.zero_grad()  # Don't change gradient on validation
                tepoch.set_description(
                    'Validate Unsupervised Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item())

    if path is not None:
        torch.save(autoencoder.state_dict(), path + '/unsupervised.model')


def test(images, path, dev='cuda', focal=False):
    device = torch.device(dev)

    autoencoder = WNet().to(device)
    autoencoder.load_state_dict(torch.load(path + '/unsupervised.model'))
    autoencoder.eval()

    max_indicies = []

    for image in images:
        image = image.to(device)
        output_tensor = autoencoder.forward_encoder(torch.unsqueeze(image, 0))
        _, max_index = torch.max(output_tensor, axis=1)
        max_indicies.append(torch.squeeze(max_index, 0))

    return transform.convert_multi_labels(max_indicies)


def report(predicts, truths):
    white_color = torch.Tensor(transform.WHITE)
    red_color = torch.Tensor(transform.RED)
    green_color = torch.Tensor(transform.GREEN)
    black_color = torch.Tensor(transform.BLACK)
    labels = [white_color, red_color, green_color, black_color]
    ious = metrics.average_iou(predicts, truths, labels)
    nonan_ious = []

    for i, iou in enumerate(ious):
        # verbose
        # print("image {0:d} IOU {1:.2f}".format(i, iou))
        if iou >= 0:
            nonan_ious.append(iou)

    print('Average image iou: {}'.format(sum(nonan_ious) / len(nonan_ious)))
    return nonan_ious


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', choices=['train', 'test', 'predict'],
    default='train')
parser.add_argument('--data-dir', required=True,
                    help='Directory that contains all labels and images')
parser.add_argument('--model-dir', required=False,
                    help='Directory that contains all models')
parser.add_argument('--output-images', action='store_true', default=False,
                    help='Output predicted images slices')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch-size', type=int, default=8)
args = parser.parse_args()

images, image_file_names = images_io.read_images(args.data_dir + '/images')

images = transform.transpose(transform.normalize(images))

if args.mode == 'train':
    multi_labels = None
    convnet = train(images, args.batch_size, args.epochs, path=args.model_dir)
elif args.mode == 'predict':
    labels, label_file_names = images_io.read_images(args.data_dir + '/labels')
    output_labels = test(images, args.model_dir, args.device)
    ious = report(output_labels, labels)
    images_io.write_images('./', output_labels, delete=False)
    print('ious: {}'.format(ious))
