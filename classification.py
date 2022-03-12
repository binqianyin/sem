import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import images_io
import transform
import metrics

'''
https://github.com/milesial/Pytorch-UNet/tree/master/unet
'''


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, is_train=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_train = is_train

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.is_train:
            # BCE loss with logits to increase numerical stability
            return logits
        else:
            return self.act(logits)


class LongUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, is_train=False):
        super(LongUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_train = is_train

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down5 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x4)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x3)
        x = self.up5(x, x1)
        logits = self.outc(x)
        if self.is_train:
            # BCE loss with logits to increase numerical stability
            return logits
        else:
            return self.act(logits)


class ThinUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ThinUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.act(logits)


class ConvNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_channels, out_channels=8, kernel_size=3, stride=1, padding=1),  # 128x128x16
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),  # 128x128x16
                                   nn.ReLU(True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LinearNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ConvNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.linear1 = nn.Linear(n_channels, 128)
        self.linear2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class UNetMulti(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, is_train=False):
        super(UNetMulti, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_train = is_train

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.act = nn.Softmax()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.is_train:
            return logits
        else:
            return self.act(logits)


def train(images, white_labels, red_labels, green_labels, black_labels, batch_size, num_epochs, path=None, fine_tune=False, focal=False, alpha=0.5, exp_dict: metrics.ExpDict = None, gamma=2.0):
    if exp_dict is not None:
        print('focal: {}, alpha: {}, gamma: {}'.format(focal, alpha, gamma))

    device = torch.device('cuda')

    red_unet = UNet(3, 1, is_train=True).to(device)
    green_unet = UNet(3, 1, is_train=True).to(device)
    black_unet = UNet(3, 1, is_train=True).to(device)
    white_unet = UNet(3, 1, is_train=True).to(device)

    if fine_tune is True:
        white_unet.load_state_dict(torch.load(path + '/white.model'))
        red_unet.load_state_dict(torch.load(path + '/red.model'))
        green_unet.load_state_dict(torch.load(path + '/green.model'))
        black_unet.load_state_dict(torch.load(path + '/black.model'))

    net_names = ['white', 'red', 'green', 'black']
    nets = [white_unet, red_unet, green_unet, black_unet]
    labels = [white_labels, red_labels, green_labels, black_labels]

    if focal is True:
        criterion = metrics.FocalDiceLoss(alpha=alpha, gamma=gamma).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)

    valid_size = len(images) // 5
    train_size = int(len(images) - valid_size)

    for net_index, net in enumerate(nets):
        dataset = images_io.ImageDataset(images, labels[net_index])
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size])
        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        if fine_tune is True:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        else:
            optimizer = torch.optim.Adam(
                net.parameters(), lr=0.001, weight_decay=1e-8)

        for epoch in range(num_epochs):
            net.train()
            with tqdm(train_data_loader) as tepoch:
                # For each batch in the dataloader
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(device)
                    color_labels = data[1].to(device)
                    output = net(color_images)
                    loss = criterion(output, color_labels)
                    iou = metrics.binary_iou(
                        torch.sigmoid(output) > 0.5, color_labels.type(torch.BoolTensor).to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_description(
                        'Train {} Epoch {}'.format(net_names[net_index], epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou.item())
                    if exp_dict is not None:
                        exp_name = '{}_focal_{}_alpha_{}_gamma_{}'.format(
                            net_names[net_index], focal, alpha, gamma)
                        exp_dict.add_metric(
                            exp_name=exp_name, entry_name='train_loss', metric=loss.item())
                        exp_dict.add_metric(
                            exp_name=exp_name, entry_name='train_iou', metric=iou.item())

            net.eval()
            with tqdm(valid_data_loader) as tepoch:
                # Validation
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(device)
                    color_labels = data[1].to(device)
                    output = net(color_images)
                    loss = criterion(output, color_labels)
                    iou = metrics.binary_iou(
                        torch.sigmoid(output) > 0.5, color_labels.type(torch.BoolTensor).to(device))
                    tepoch.set_description(
                        'Validate {} Epoch {}'.format(net_names[net_index], epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou.item())
                    if exp_dict is not None:
                        exp_name = '{}_focal_{}_alpha_{}_gamma_{}'.format(
                            net_names[net_index], focal, alpha, gamma)
                        exp_dict.add_metric(
                            exp_name=exp_name, entry_name='validate_loss', metric=loss.item())
                        exp_dict.add_metric(
                            exp_name=exp_name, entry_name='validate_iou', metric=iou.item())

        if path is not None:
            torch.save(net.state_dict(), path + '/' +
                       net_names[net_index] + '.model')

    return red_unet, green_unet, black_unet


def train_synthesizer(images, multi_labels, batch_size, epochs, path=None, fine_tune=False, focal=False, exp_dict=None):
    if exp_dict is not None:
        print('focal: {}'.format(focal))

    device = torch.device('cuda')

    if focal is True:
        criterion = metrics.FocalDiceLossMulti(weight=None).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    valid_size = len(images) // 5
    train_size = int(len(images) - valid_size)

    dataset = images_io.ImageDataset(images, multi_labels)
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    convnet = UNetMulti(8, 4, is_train=True).to(device)

    if fine_tune is True:
        convnet.load_state_dict(torch.load(path + 'synthesizer.model'))

    if fine_tune is True:
        optimizer = torch.optim.Adam(convnet.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.Adam(convnet.parameters(), lr=0.001)

    # we do not specify pretrained=True, i.e. do not load default weights
    white_unet = UNet(3, 1).to(device)
    red_unet = UNet(3, 1).to(device)
    green_unet = UNet(3, 1).to(device)
    black_unet = UNet(3, 1).to(device)
    multi_unet = UNetMulti(3, 4).to(device)

    if fine_tune is True:
        white_unet.load_state_dict(torch.load(path + '/fine-tune/white.model'))
        red_unet.load_state_dict(torch.load(path + '/fine-tune/red.model'))
        green_unet.load_state_dict(torch.load(path + '/fine-tune/green.model'))
        black_unet.load_state_dict(torch.load(path + '/fine-tune/black.model'))
        multi_unet.load_state_dict(torch.load(path + '/fine-tune/multi.model'))
    else:
        white_unet.load_state_dict(torch.load(path + '/white.model'))
        red_unet.load_state_dict(torch.load(path + '/red.model'))
        green_unet.load_state_dict(torch.load(path + '/green.model'))
        black_unet.load_state_dict(torch.load(path + '/black.model'))
        multi_unet.load_state_dict(torch.load(path + '/multi.model'))

    white_unet.eval()
    red_unet.eval()
    green_unet.eval()
    black_unet.eval()
    multi_unet.eval()

    for epoch in range(epochs):
        convnet.train()
        with tqdm(train_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                color_images = data[0].to(device)
                color_labels = data[1].to(device)
                with torch.no_grad():
                    color_white = white_unet(color_images)
                    color_red = red_unet(color_images)
                    color_green = green_unet(color_images)
                    color_black = black_unet(color_images)
                    colors = multi_unet(color_images)
                    synthesized = torch.cat(
                        [color_white, color_red, color_green, color_black, colors], dim=1)
                output = convnet(synthesized)
                max_index = torch.argmax(output, dim=1)
                iou, ious = metrics.average_iou_tensor(
                    max_index, color_labels, [0, 1, 2, 3])
                loss = criterion(output, color_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_description(
                    'Train Synthesizer Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item(), iou=iou)

                if exp_dict is not None:
                    exp_name = 'focal_{}'.format(focal)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_loss', metric=loss.item())
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_iou', metric=iou)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_ious', metric=ious)

        convnet.eval()
        with tqdm(valid_data_loader) as tepoch:
            # Validation
            for _, data in enumerate(tepoch):
                color_images = data[0].to(device)
                with torch.no_grad():
                    color_white = white_unet(color_images)
                    color_red = red_unet(color_images)
                    color_green = green_unet(color_images)
                    color_black = black_unet(color_images)
                    colors = multi_unet(color_images)
                    synthesized = torch.cat(
                        [color_white, color_red, color_green, color_black, colors], dim=1)
                color_labels = data[1].to(device)
                output = convnet(synthesized)
                max_index = torch.argmax(output, dim=1)
                iou, ious = metrics.average_iou_tensor(
                    max_index, color_labels, [0, 1, 2, 3])
                loss = criterion(output, color_labels)
                tepoch.set_description(
                    'Validate Synthesizer Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item(), iou=iou)

                if exp_dict is not None:
                    exp_name = 'focal_{}'.format(focal)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_loss', metric=loss.item())
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_iou', metric=iou)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_ious', metric=ious)

    if path is not None:
        if fine_tune is True:
            torch.save(convnet.state_dict(), path +
                       '/fine-tune/synthesizer.model')
        else:
            torch.save(convnet.state_dict(), path + '/synthesizer.model')


def train_multi(images, multi_labels, batch_size, epochs, focal=False, path=None, weight=False, exp_dict=None):
    if exp_dict is not None:
        print('focal: {}, weight: {}'.format(focal, weight))

    device = torch.device('cuda')

    valid_size = len(images) // 5
    train_size = int(len(images) - valid_size)

    if focal is True:
        if weight is True:
            focal_weight = metrics.color_weight(multi_labels).to(device)
        else:
            focal_weight = None
        # criterion = metrics.FocalLossMulti(
        #    weight=focal_weight, gamma=2.0).to(device)
        criterion = metrics.FocalDiceLossMulti(weight=focal_weight).to(device)
    else:
        criterion = metrics.FocalLossMulti(
            weight=None, gamma=2.0).to(device)
        #criterion = nn.CrossEntropyLoss().to(device)

    dataset = images_io.ImageDataset(images, multi_labels)
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    convnet = UNetMulti(3, 4, is_train=True).to(device)
    optimizer = torch.optim.Adam(
        convnet.parameters(), lr=0.001, weight_decay=1e-8)

    for epoch in range(epochs):
        convnet.train()
        with tqdm(train_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                color_images = data[0].to(device)
                color_labels = data[1].to(device)
                output = convnet(color_images)
                max_index = torch.argmax(output, dim=1)
                iou, ious = metrics.average_iou_tensor(
                    max_index, color_labels, [0, 1, 2, 3])
                loss = criterion(output, color_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tepoch.set_description(
                    'Train Multi Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item(), iou=iou)

                if exp_dict is not None:
                    exp_name = 'focal_{}_weight_{}'.format(focal, weight)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_loss', metric=loss.item())
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_iou', metric=iou)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='train_ious', metric=ious)

        convnet.eval()
        with tqdm(valid_data_loader) as tepoch:
            # Validation
            for _, data in enumerate(tepoch):
                color_images = data[0].to(device)
                color_labels = data[1].to(device)
                output = convnet(color_images)
                max_index = torch.argmax(output, dim=1)
                iou, ious = metrics.average_iou_tensor(
                    max_index, color_labels, [0, 1, 2, 3])
                loss = criterion(output, color_labels)
                tepoch.set_description(
                    'Validate Multi Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item(), iou=iou)

                if exp_dict is not None:
                    exp_name = 'focal_{}_weight_{}'.format(focal, weight)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_loss', metric=loss.item())
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_iou', metric=iou)
                    exp_dict.add_metric(
                        exp_name=exp_name, entry_name='validate_ious', metric=ious)

    if path is not None:
        torch.save(convnet.state_dict(), path + '/multi.model')


def test(images, path, dev='cuda'):
    device = torch.device(dev)

    # we do not specify pretrained=True, i.e. do not load default weights
    white_unet = UNet(3, 1).to(device)
    red_unet = UNet(3, 1).to(device)
    green_unet = UNet(3, 1).to(device)
    black_unet = UNet(3, 1).to(device)
    white_unet.load_state_dict(torch.load(path + '/' + 'white.model'))
    red_unet.load_state_dict(torch.load(path + '/' + 'red.model'))
    green_unet.load_state_dict(torch.load(
        path + '/' + 'green.model'))
    black_unet.load_state_dict(torch.load(
        path + '/' + 'black.model'))

    nets = [white_unet, red_unet, green_unet, black_unet]
    output_labels = []

    for image in images:
        image = image.to(device)
        outputs = []
        for net in nets:
            net.eval()
            output = net(image.unsqueeze(0)).squeeze(0).squeeze(0)
            outputs.append(output)
        output_tensor = torch.dstack(outputs)
        max_values, max_index = torch.max(output_tensor, axis=-1)
        output_labels += transform.convert_labels(
            [max_values], [max_index])

    return output_labels


def test_synthesizer(images, path, dev='cuda'):
    device = torch.device(dev)

    # we do not specify pretrained=True, i.e. do not load default weights
    white_unet = UNet(3, 1).to(device)
    red_unet = UNet(3, 1).to(device)
    green_unet = UNet(3, 1).to(device)
    black_unet = UNet(3, 1).to(device)
    multi_unet = UNetMulti(3, 4).to(device)
    white_unet.load_state_dict(torch.load(path + '/white.model'))
    red_unet.load_state_dict(torch.load(path + '/red.model'))
    green_unet.load_state_dict(torch.load(path + '/green.model'))
    black_unet.load_state_dict(torch.load(path + '/black.model'))
    multi_unet.load_state_dict(torch.load(path + '/multi.model'))
    white_unet.eval()
    red_unet.eval()
    green_unet.eval()
    black_unet.eval()
    multi_unet.eval()

    convnet = UNetMulti(8, 4).to(device)
    convnet.load_state_dict(torch.load(path + '/synthesizer.model'))
    convnet.eval()

    max_indicies = []

    for image in images:
        image = image.to(device)
        color_white = white_unet(torch.unsqueeze(image, 0))
        color_red = red_unet(torch.unsqueeze(image, 0))
        color_green = green_unet(torch.unsqueeze(image, 0))
        color_black = black_unet(torch.unsqueeze(image, 0))
        colors = multi_unet(torch.unsqueeze(image, 0))
        synthesized = torch.cat(
            [color_white, color_red, color_green, color_black, colors], dim=1)
        output_tensor = convnet(synthesized)
        _, max_index = torch.max(output_tensor, axis=1)
        max_indicies.append(torch.squeeze(max_index, 0))

    return transform.convert_multi_labels(max_indicies)


def test_multi(images, path, dev='cuda', focal=False):
    device = torch.device(dev)

    convnet = UNetMulti(3, 4).to(device)
    convnet.load_state_dict(torch.load(path + '/multi.model'))
    convnet.eval()

    max_indicies = []

    for image in images:
        image = image.to(device)
        output_tensor = convnet(torch.unsqueeze(image, 0))
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


def output_fake(images, labels, ious, fake_threshold, real_threshold, path):
    real_images = []
    real_image_labels = []
    fake_images = []
    fake_image_labels = []
    for i, iou in enumerate(ious):
        image = images[i]
        label = labels[i]
        if iou <= fake_threshold:
            fake_images.append(image)
            fake_image_labels.append(label)
        elif iou >= real_threshold:
            real_images.append(image)
            real_image_labels.append(label)
    real_images = transform.denormalize(
        transform.transpose(real_images, False))
    fake_images = transform.denormalize(
        transform.transpose(fake_images, False))
    images_io.write_images(path + '/real/images', real_images)
    images_io.write_images(path + '/real/labels', real_image_labels)
    images_io.write_images(path + '/fake/images', fake_images)
    images_io.write_images(path + '/fake/labels', fake_image_labels)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', choices=['train', 'train-synthesizer', 'train-multi',
                       'test', 'test-synthesizer', 'test-multi',
                       'predict', 'predict-synthesizer', 'predict-multi'],
    default='train')
parser.add_argument('--fine-tune', action='store_true', default=False,
                    help='Fine-tune based on trained models')
parser.add_argument('--fine-tune-with-fake', action='store_true', default=False,
                    help='Fine-tune based on trained models with fake images only')
parser.add_argument('--data-dir', required=True,
                    help='Directory that contains all labels and images')
parser.add_argument('--model-dir', required=False,
                    help='Directory that contains all models')
parser.add_argument('--output-images', action='store_true', default=False,
                    help='Output predicted images slices')
parser.add_argument('--output-fake', action='store_true', default=False,
                    help='Output fake and real images to data_dir')
parser.add_argument('--fake-threshold', type=float, default=0.4,
                    help='Lowest IOU to identify a real image')
parser.add_argument('--real-threshold', type=float, default=0.45,
                    help='Highest IOU to identify a fake image')
parser.add_argument('--focal', action='store_true', default=False)
parser.add_argument('--weight', action='store_true', default=False)
parser.add_argument('--experiments', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

if args.fine_tune_with_fake is True:
    fake_images, _ = images_io.read_images(args.data_dir + '/fake/images')
    fake_labels, _ = images_io.read_images(args.data_dir + '/fake/labels')
    real_images, _ = images_io.read_images(args.data_dir + '/fake/images')
    real_labels, _ = images_io.read_images(args.data_dir + '/fake/labels')
    images = real_images[0: len(fake_images)] + fake_images
    labels = real_labels[0: len(fake_labels)] + fake_labels
else:
    images, image_file_names = images_io.read_images(args.data_dir + '/images')
    labels, label_file_names = images_io.read_images(args.data_dir + '/labels')

images = transform.transpose(transform.normalize(images))

fine_tune = args.fine_tune or args.fine_tune_with_fake

if args.mode == 'train':
    white_labels, red_labels, green_labels, black_labels = transform.prepare_binary_labels(
        labels)
    white_labels = transform.transpose(white_labels)
    red_labels = transform.transpose(red_labels)
    green_labels = transform.transpose(green_labels)
    black_labels = transform.transpose(black_labels)
    if args.experiments:
        exp_dict = metrics.ExpDict('train.json')
        train(images, white_labels, red_labels, green_labels, black_labels, args.batch_size,
              args.epochs, path=args.model_dir, fine_tune=fine_tune, focal=True, exp_dict=exp_dict)
        exp_dict.dump(args.model_dir)
    else:
        train(images, white_labels, red_labels, green_labels, black_labels, args.batch_size,
              args.epochs, path=args.model_dir, fine_tune=fine_tune, focal=args.focal)
elif args.mode == 'train-synthesizer':
    multi_labels = transform.prepare_multi_labels(labels)
    if args.experiments:
        exp_dict = metrics.ExpDict('train-synthesizer.json')
        train_synthesizer(images, multi_labels, args.batch_size, args.epochs,
                          path=args.model_dir, fine_tune=fine_tune, focal=args.focal, exp_dict=exp_dict)
        exp_dict.dump(args.model_dir)
    else:
        convnet = train_synthesizer(
            images, multi_labels, args.batch_size, args.epochs, path=args.model_dir, fine_tune=fine_tune)
elif args.mode == 'train-multi':
    multi_labels = transform.prepare_multi_labels(labels)
    if args.experiments:
        exp_dict = metrics.ExpDict('train-multi.json')
        convnet = train_multi(images, multi_labels, args.batch_size, args.epochs,
                              path=args.model_dir, focal=args.focal, weight=args.weight, exp_dict=exp_dict)
        exp_dict.dump(args.model_dir)
    else:
        convnet = train_multi(
            images, multi_labels, args.batch_size, args.epochs, focal=args.focal, path=args.model_dir)
elif args.mode == 'test':
    output_labels = test(images, args.model_dir)
    ious = report(output_labels, labels)
    if args.output_images:
        images_io.write_images(args.data_dir + '/predicts', output_labels)
    if args.output_fake:
        output_fake(images, labels, ious, args.fake_threshold,
                    args.real_threshold, args.data_dir)
elif args.mode == 'test-synthesizer':
    output_labels = test_synthesizer(images, args.model_dir)
    ious = report(output_labels, labels)
    if args.output_images:
        images_io.write_images(args.data_dir + '/predicts', output_labels)
    if args.output_fake:
        output_fake(images, labels, ious, args.fake_threshold,
                    args.real_threshold, args.data_dir)
elif args.mode == 'test-multi':
    output_labels = test_multi(images, args.model_dir, focal=args.focal)
    ious = report(output_labels, labels)
    if args.output_images:
        images_io.write_images(args.data_dir + '/predicts', output_labels)
    if args.output_fake:
        output_fake(images, labels, ious, args.fake_threshold,
                    args.real_threshold, args.data_dir)
elif args.mode == 'predict':
    output_labels = test(images, args.model_dir, args.device)
    ious = report(output_labels, labels)
    images_io.write_images('./', output_labels, delete=False)
    print('ious: {}'.format(ious))
elif args.mode == 'predict-synthesizer':
    output_labels = test_synthesizer(images, args.model_dir, args.device)
    ious = report(output_labels, labels)
    images_io.write_images('./', output_labels, delete=False)
    print('ious: {}'.format(ious))
elif args.mode == 'predict-multi':
    output_labels = test_multi(images, args.model_dir, args.device)
    ious = report(output_labels, labels)
    images_io.write_images('./', output_labels, delete=False)
    print('ious: {}'.format(ious))
