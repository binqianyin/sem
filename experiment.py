
from abc import ABC

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import images_io
import metric
import network
import transform


class Experiment(ABC):
    '''
        An abstract class for training and testing
    '''

    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.exp_dict = metric.ExpDict(f'{name}.json')

    def prepare_data(self, data_dir):
        '''
          Prepare data and labels for training and testing

          :param data_dir: path to the data
        '''
        self.images, _ = images_io.read_images(data_dir + '/images')
        self.labels, _ = images_io.read_images(data_dir + '/labels')

        self.images = transform.transpose(transform.normalize(self.images))

    def update_exp_dict(self, exp_name, mode, loss=None,
                        iou=None, ious=None, **kwargs):
        '''
            Update the experiment dictionary

            :param exp_name: The name of the experiment
            :param mode: The mode of the experiment
            :param loss: The loss of the experiment
            :param iou: The IoU of the experiment
            :param ious: The IoUs of the experiment
            :param kwargs: The other metrics of the experiment
        '''
        if 'focal' in self.exp_dict and self.exp_dict['focal']:
            exp_name += '_focal'
        if 'alpha' in self.exp_dict and self.exp_dict['alpha']:
            exp_name += '_alpha'
        if 'gamma' in self.exp_dict and self.exp_dict['gamma']:
            exp_name += '_gamma'
        if loss is not None:
            self.exp_dict.add_metric(
                exp_name=exp_name, entry_name=f'{mode}_loss', metric=loss)
        if iou is not None:
            self.exp_dict.add_metric(
                exp_name=exp_name, entry_name=f'{mode}_iou', metric=iou)
        if ious is not None:
            self.exp_dict.add_metric(
                exp_name=exp_name, entry_name=f'{mode}_ious', metric=ious)

    def report(self, predicts, truths):
        '''
          Report the results of the experiment

          :param predicts: predicted labels
          :param truths: true labels
        '''
        white_color = torch.Tensor(transform.WHITE)
        red_color = torch.Tensor(transform.RED)
        green_color = torch.Tensor(transform.GREEN)
        black_color = torch.Tensor(transform.BLACK)
        labels = [white_color, red_color, green_color, black_color]
        ious = metric.average_iou(predicts, truths, labels)
        nonan_ious = []

        for i, iou in enumerate(ious):
            # throw out nan values
            print("image {0:d} IOU {1:.2f}".format(i, iou))
            if iou >= 0:
                nonan_ious.append(iou)

        print(
            'Average image iou: {}'.format(
                sum(nonan_ious) /
                len(nonan_ious)))
        return nonan_ious

    def predict(self, model_dir, output_dir):
        '''
          Predict using the trained model
        '''
        output_labels = self.test(model_dir)
        self.report(output_labels, self.labels)
        if output_dir is not None:
            images_io.write_images(output_dir, output_labels)

    def train(self, batch_size, num_epochs, model_dir, **kwargs):
        '''
          Training models with given hyperparameters

          :param batch_size: batch size for training
          :param num_epochs: number of epochs for training
          :kwargs: other hyperparameters

          :return: trained models
        '''
        raise NotImplementedError

    def test(self, model_dir):
        '''
          Test the trained model

          :param path: path to the trained model
        '''
        raise NotImplementedError


class Backbone(Experiment):
    '''
      The backbone model reads in images of different colors and trains individual models for each color.
    '''

    def __init__(self, device):
        super().__init__('backbone', device)

    def prepare_data(self, path):
        super().prepare_data(path)
        self.white_labels, self.red_labels, self.green_labels, self.black_labels = \
            transform.prepare_binary_labels(self.labels)
        self.white_labels = transform.transpose(self.white_labels)
        self.red_labels = transform.transpose(self.red_labels)
        self.green_labels = transform.transpose(self.green_labels)
        self.black_labels = transform.transpose(self.black_labels)

    def train(self, batch_size, num_epochs, model_dir, **kwargs):
        # Init models of different colors
        red_unet = network.UNet(3, 1, is_train=True).to(self.device)
        green_unet = network.UNet(3, 1, is_train=True).to(self.device)
        black_unet = network.UNet(3, 1, is_train=True).to(self.device)
        white_unet = network.UNet(3, 1, is_train=True).to(self.device)

        net_names = ['white', 'red', 'green', 'black']
        nets = [white_unet, red_unet, green_unet, black_unet]
        labels = [self.white_labels, self.red_labels,
                  self.green_labels, self.black_labels]

        # Select which loss function to use
        if 'focal' in kwargs and kwargs['focal']:
            criterion = metric.FocalDiceLoss(
                alpha=kwargs['alpha'], gamma=kwargs['gamma']).to(self.device)
        else:
            criterion = nn.BCEWithLogitsLoss().to(self.device)

        valid_size = len(self.images) // 5
        train_size = int(len(self.images) - valid_size)

        for net_index, net in enumerate(nets):
            dataset = images_io.ImageDataset(self.images, labels[net_index])
            train_dataset, valid_dataset = random_split(
                dataset, [train_size, valid_size])
            train_data_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            valid_data_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=0.001, weight_decay=1e-8)

            for epoch in range(num_epochs):
                net.train()
                with tqdm(train_data_loader) as tepoch:
                    # For each batch in the dataloader
                    for _, data in enumerate(tepoch):
                        color_images = data[0].to(self.device)
                        color_labels = data[1].to(self.device)
                        output = net(color_images)
                        loss = criterion(output, color_labels)
                        # 0.5 is used as the sigmod threshold
                        iou = metric.binary_iou(
                            torch.sigmoid(output) > 0.5, color_labels.type(torch.BoolTensor).to(self.device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        tepoch.set_description(
                            'Train {} Epoch {}'.format(net_names[net_index], epoch))
                        tepoch.set_postfix(loss=loss.item(), iou=iou)
                        self.update_exp_dict(
                            net_names[net_index], mode='train', loss=loss.item(), iou=iou, **kwargs)

                net.eval()
                with tqdm(valid_data_loader) as tepoch:
                    # Validation
                    for _, data in enumerate(tepoch):
                        color_images = data[0].to(self.device)
                        color_labels = data[1].to(self.device)
                        output = net(color_images)
                        loss = criterion(output, color_labels)
                        iou = metric.binary_iou(
                            torch.sigmoid(output) > 0.5, color_labels.type(torch.BoolTensor).to(self.device))
                        tepoch.set_description(
                            'Validate {} Epoch {}'.format(net_names[net_index], epoch))
                        tepoch.set_postfix(loss=loss.item(), iou=iou)
                        self.update_exp_dict(
                            net_names[net_index], mode='validate', loss=loss.item(), iou=iou, **kwargs)

            torch.save(net.state_dict(), model_dir + '/' +
                       net_names[net_index] + '.model')

        self.exp_dict.dump(model_dir)
        return red_unet, green_unet, black_unet

    def test(self, model_dir):
        # we do not specify pretrained=True, i.e. do not load default weights
        white_unet = network.UNet(3, 1).to(self.device)
        red_unet = network.UNet(3, 1).to(self.device)
        green_unet = network.UNet(3, 1).to(self.device)
        black_unet = network.UNet(3, 1).to(self.device)
        white_unet.load_state_dict(torch.load(model_dir + '/' + 'white.model'))
        red_unet.load_state_dict(torch.load(model_dir + '/' + 'red.model'))
        green_unet.load_state_dict(torch.load(
            model_dir + '/' + 'green.model'))
        black_unet.load_state_dict(torch.load(
            model_dir + '/' + 'black.model'))

        nets = [white_unet, red_unet, green_unet, black_unet]
        output_labels = []

        for image in self.images:
            # For each image, output the label according to the maximum
            # probability of each pixel
            image = image.to(self.device)
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


class Synthesizer(Experiment):
    def __init__(self, device):
        super().__init__('synthesizer', device)

    def prepare_data(self, path):
        super().prepare_data(path)
        self.multi_labels = transform.prepare_multi_labels(self.labels)

    def train(self, batch_size, num_epochs, model_dir, **kwargs):
        if 'focal' in kwargs and kwargs['focal']:
            criterion = metric.FocalDiceLossMulti(weight=None).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss().to(self.device)

        valid_size = len(self.images) // 5
        train_size = int(len(self.images) - valid_size)

        dataset = images_io.ImageDataset(self.images, self.multi_labels)
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size])
        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        convnet = network.UNetMulti(8, 4, is_train=True).to(self.device)

        optimizer = torch.optim.Adam(convnet.parameters(), lr=0.001)

        # we do not specify pretrained=True, i.e. do not load default weights
        white_unet = network.UNet(3, 1).to(self.device)
        red_unet = network.UNet(3, 1).to(self.device)
        green_unet = network.UNet(3, 1).to(self.device)
        black_unet = network.UNet(3, 1).to(self.device)
        multi_unet = network.UNetMulti(3, 4).to(self.device)

        white_unet.load_state_dict(torch.load(model_dir + '/white.model'))
        red_unet.load_state_dict(torch.load(model_dir + '/red.model'))
        green_unet.load_state_dict(torch.load(model_dir + '/green.model'))
        black_unet.load_state_dict(torch.load(model_dir + '/black.model'))
        multi_unet.load_state_dict(torch.load(model_dir + '/multi.model'))

        white_unet.eval()
        red_unet.eval()
        green_unet.eval()
        black_unet.eval()
        multi_unet.eval()

        for epoch in range(num_epochs):
            convnet.train()
            with tqdm(train_data_loader) as tepoch:
                # For each batch in the dataloader
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(self.device)
                    color_labels = data[1].to(self.device)
                    # Run inference of each network, do not record the gradients since
                    # we don't update the weights of those networks
                    with torch.no_grad():
                        color_white = white_unet(color_images)
                        color_red = red_unet(color_images)
                        color_green = green_unet(color_images)
                        color_black = black_unet(color_images)
                        colors = multi_unet(color_images)
                        synthesized = torch.cat(
                            [color_white, color_red, color_green, color_black, colors], dim=1)
                    # convent needs weight update
                    output = convnet(synthesized)
                    max_index = torch.argmax(output, dim=1)
                    iou, ious = metric.average_iou_tensor(
                        max_index, color_labels, [0, 1, 2, 3])
                    loss = criterion(output, color_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_description(
                        'Train Synthesizer Epoch {}'.format(epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou)
                    self.update_exp_dict(
                        self.name, 'train', loss.item(), iou, ious, **kwargs)

            convnet.eval()
            with tqdm(valid_data_loader) as tepoch:
                # Validation
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(self.device)
                    with torch.no_grad():
                        color_white = white_unet(color_images)
                        color_red = red_unet(color_images)
                        color_green = green_unet(color_images)
                        color_black = black_unet(color_images)
                        colors = multi_unet(color_images)
                        synthesized = torch.cat(
                            [color_white, color_red, color_green, color_black, colors], dim=1)
                    color_labels = data[1].to(self.device)
                    output = convnet(synthesized)
                    max_index = torch.argmax(output, dim=1)
                    iou, ious = metric.average_iou_tensor(
                        max_index, color_labels, [0, 1, 2, 3])
                    loss = criterion(output, color_labels)
                    tepoch.set_description(
                        'Validate Synthesizer Epoch {}'.format(epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou)
                    self.update_exp_dict(
                        self.name, 'validate', loss.item(), iou, ious, **kwargs)

        self.exp_dict.dump(model_dir)
        torch.save(convnet.state_dict(), model_dir + '/synthesizer.model')

        return convnet

    def test(self, model_dir):
        # we do not specify pretrained=True, i.e. do not load default weights
        white_unet = network.UNet(3, 1).to(self.device)
        red_unet = network.UNet(3, 1).to(self.device)
        green_unet = network.UNet(3, 1).to(self.device)
        black_unet = network.UNet(3, 1).to(self.device)
        multi_unet = network.UNetMulti(3, 4).to(self.device)
        white_unet.load_state_dict(torch.load(model_dir + '/white.model'))
        red_unet.load_state_dict(torch.load(model_dir + '/red.model'))
        green_unet.load_state_dict(torch.load(model_dir + '/green.model'))
        black_unet.load_state_dict(torch.load(model_dir + '/black.model'))
        multi_unet.load_state_dict(torch.load(model_dir + '/multi.model'))
        white_unet.eval()
        red_unet.eval()
        green_unet.eval()
        black_unet.eval()
        multi_unet.eval()

        convnet = network.UNetMulti(8, 4).to(self.device)
        convnet.load_state_dict(torch.load(model_dir + '/synthesizer.model'))
        convnet.eval()

        max_indicies = []

        for image in self.images:
            # For each image in the dataset, run multiple networks and combine
            # the results as the input to the synthesizer
            image = image.to(self.device)
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


class Multiclass(Experiment):
    def __init__(self, device):
        super().__init__('multi', device)

    def prepare_data(self, data_dir):
        super().prepare_data(data_dir)
        self.multi_labels = transform.prepare_multi_labels(self.labels)

    def train(self, batch_size, num_epochs, model_dir, **kwargs):
        valid_size = len(self.images) // 5
        train_size = int(len(self.images) - valid_size)

        if 'focal' in kwargs and kwargs['focal']:
            criterion = metric.FocalDiceLossMulti().to(self.device)
        else:
            criterion = metric.FocalLossMulti(
                weight=None, gamma=2.0).to(self.device)

        dataset = images_io.ImageDataset(self.images, self.multi_labels)
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size])
        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        convnet = network.UNetMulti(3, 4, is_train=True).to(self.device)
        optimizer = torch.optim.Adam(
            convnet.parameters(), lr=0.001, weight_decay=1e-8)

        for epoch in range(num_epochs):
            convnet.train()
            with tqdm(train_data_loader) as tepoch:
                # For each batch in the dataloader
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(self.device)
                    color_labels = data[1].to(self.device)
                    output = convnet(color_images)
                    max_index = torch.argmax(output, dim=1)
                    iou, ious = metric.average_iou_tensor(
                        max_index, color_labels, [0, 1, 2, 3])
                    loss = criterion(output, color_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_description(
                        'Train Multi Epoch {}'.format(epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou)
                    self.update_exp_dict(
                        self.name, 'train', loss.item(), iou, ious, **kwargs)

            convnet.eval()
            with tqdm(valid_data_loader) as tepoch:
                # Validation
                for _, data in enumerate(tepoch):
                    color_images = data[0].to(self.device)
                    color_labels = data[1].to(self.device)
                    output = convnet(color_images)
                    max_index = torch.argmax(output, dim=1)
                    iou, ious = metric.average_iou_tensor(
                        max_index, color_labels, [0, 1, 2, 3])
                    loss = criterion(output, color_labels)
                    tepoch.set_description(
                        'Validate Multi Epoch {}'.format(epoch))
                    tepoch.set_postfix(loss=loss.item(), iou=iou)
                    self.update_exp_dict(
                        self.name, 'validate', loss.item(), iou, ious, **kwargs)

        self.exp_dict.dump(model_dir)
        torch.save(convnet.state_dict(), model_dir + '/multi.model')

        return convnet

    def test(self, model_dir):
        convnet = network.UNetMulti(3, 4).to(self.device)
        convnet.load_state_dict(torch.load(model_dir + '/multi.model'))
        convnet.eval()

        max_indicies = []

        for image in self.images:
            # For each image, get the max index of the output
            image = image.to(self.device)
            output_tensor = convnet(torch.unsqueeze(image, 0))
            _, max_index = torch.max(output_tensor, axis=1)
            max_indicies.append(torch.squeeze(max_index, 0))

        return transform.convert_multi_labels(max_indicies)
