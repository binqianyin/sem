import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import images_io
import transform
import metrics


# TODO(Keren): shallow GAN
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def train(images, labels, batch_size, num_epochs, path=None):
    # Config traning parameters
    valid_size = len(images) // 5
    train_size = int(len(images) - valid_size)
    dataset = images_io.ImageDataset(images, labels)
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    device = torch.device('cuda')

    discrimnator = Discriminator(3, 16).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        discrimnator.parameters(), lr=0.01, amsgrad=True)

    for epoch in range(num_epochs):
        discrimnator.train()
        with tqdm(train_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                ############################
                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with a mixture of real and fake batch
                # Format batch
                train_images = data[0].to(device)
                train_labels = data[1].to(device)
                # Forward pass real batch through D
                output = discrimnator(train_images).view(-1)
                # Calculate loss on all-real batch
                loss = criterion(output, train_labels)
                # Calculate gradients for D in backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_description(
                    'Train Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item())

        discrimnator.eval()
        with tqdm(valid_data_loader) as tepoch:
            # For each batch in the dataloader
            for _, data in enumerate(tepoch):
                # Format batch
                validate_images = data[0].to(device)
                validate_labels = data[1].to(device)
                output = discrimnator(validate_images).view(-1)
                loss = criterion(output, validate_labels)

                acc = metrics.accuracy(output, validate_labels)

                tepoch.set_description('Validate Epoch {}'.format(epoch))
                tepoch.set_postfix(loss=loss.item(), acc=acc.item())

    # Final test
    discrimnator.eval()
    accs = []
    # For each batch in the dataloader
    for _, data in enumerate(valid_data_loader):
        # Format batch
        validate_images = data[0].to(device)
        validate_labels = data[1].to(device)
        output = discrimnator(validate_images).view(-1)
        loss = criterion(output, validate_labels)

        accs.append(metrics.accuracy(output, validate_labels))
    print('Test accuracy {0:.2f}'.format(sum(accs) / len(accs)))

    if path is not None:
        torch.save(discrimnator.state_dict(), path + '/discriminator.model')


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--data-dir', required=True,
                    help='Directory that contains all labels and images')
parser.add_argument('--model-dir', required=False,
                    help='Directory that contains all models')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()

real_images, _ = images_io.read_images(args.data_dir + '/real/images')
fake_images, _ = images_io.read_images(args.data_dir + '/fake/images')

real_images = transform.transpose(transform.normalize(real_images))
fake_images = transform.transpose(transform.normalize(fake_images))

labels = torch.hstack([torch.ones(len(real_images)),
                      torch.zeros(len(fake_images))])
images = real_images + fake_images

if args.mode == 'train':
    train(images, labels, args.batch_size, args.epochs, args.model_dir)
elif args.mode == 'test':
    pass
