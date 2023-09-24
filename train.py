import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data import CIFAR10Policy, Cutout
from data.sampler import DistributedSampler

import time
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from models import *
from models.vit import VisionTransformer
from models.resnet import resnet20_v2_cifar_modified
from models.spike_model import SpikeModel
from IPython import embed

_seed_ = 3407
import random

random.seed(3407)

torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./cifar10/',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='./cifar10/',
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./cifar100/',
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root='./cifar100/',
                               train=False, download=False, transform=transform_test)

    # train_sampler = DistributedSampler(train_dataset)
    # val_sampler = DistributedSampler(val_dataset, round_up=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res20_m2', type=str, help='model name',
                        choices=['vit', 'res20_m2'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=128, type=int, help='snn threshold')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--shift_snn', default=100, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--spike', action='store_false', help='use spiking network')
    parser.add_argument('--teacher', action='store_true', help='use teacher')
    parser.add_argument('--rp', action='store_true', help='use teacher')
    parser.add_argument('--recon', action='store_true', help='use teacher')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight', type=float, default=0.1, help='weight for kd loss')

    args = parser.parse_args()

    writer = SummaryWriter(f'./summaries/{args.dataset}_{args.arch}')
    device = torch.device("cuda:4" if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    use_cifar10 = args.dataset == 'CIFAR10'

    train_loader, test_loader = build_data(cutout=True, use_cifar10=use_cifar10, auto_aug=True,
                                           batch_size=args.batch_size)
    best_acc = 0
    best_epoch = 0

    model_save_name = f'{args.arch}.pth'

    if args.arch == 'vit':
        ann = VisionTransformer(
            img_size=32,
            patch_size=8,
            in_chans=3,
            class_dim=100,
            embed_dim=252,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            epsilon=1e-5
        )
    elif args.arch == 'res20_m2':
        ann = resnet20_v2_cifar_modified(num_classes=10 if use_cifar10 else 100)

    if args.spike:
        ann = SpikeModel(ann, args.step)
    print(ann)
    ann.to(device)

    num_epochs = 1000
    criterion = nn.CrossEntropyLoss().to(device)

    parameters = split_weights(ann)
    optimizer = torch.optim.SGD(params=parameters, lr=0.1, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(ann.parameters(), lr=0.001, weight_decay=0.009)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

    correct = torch.Tensor([0.]).to(device)
    total = torch.Tensor([0.]).to(device)
    acc = torch.Tensor([0.]).to(device)

    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)

            outputs = ann(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i + 1) % 80 == 0:
                print('Time elapsed: {}'.format(time.time() - start_time))
                writer.add_scalar('Train Loss /batchidx', loss, i + len(train_loader) * epoch)
        scheduler.step()
        writer.add_scalar('Train Loss /epoch', running_loss / len(train_loader), epoch)

        correct = torch.Tensor([0.]).to(device)
        total = torch.Tensor([0.]).to(device)
        acc = torch.Tensor([0.]).to(device)

        # test
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = ann(inputs)

                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(ann.state_dict(), model_save_name)
        print('best_acc is: {}'.format(best_acc))
        print('best_iter: {}'.format(best_epoch))
        print('Iters: {}\n\n'.format(epoch))
        writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)

    writer.close()


