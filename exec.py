import torch
import torch.optim as optim
from torchvision import datasets, transforms
from cifar import resnet20
from preact import resnet18
from utils import *

cuda_available = torch.cuda.is_available()


def main(use_mixup, alpha, batch_size, share, preact):
    transform_for_mixup = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader_1 = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_for_mixup),
            batch_size=batch_size,
            shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, transform=transform_for_mixup), batch_size=batch_size,
            shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform_test), batch_size=batch_size,
            shuffle=True)

    model = resnet18(num_classes=10) if preact else resnet20()
    if cuda_available:
        model.cuda()
    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for idx in range(200):
        print(f"epoch: {idx}")
        scheduler.step()
        if use_mixup:
            mixup_train(model=model, optimizer=optimizer,
                        data_loaders=(train_loader_1, train_loader_2), alpha=alpha,
                        data_length=len(train_loader_1), share_lambda=share)
        else:
            standard_train(model=model, optimizer=optimizer,
                           data_loader=train_loader, data_length=len(train_loader))
        test(model=model, data_loader=test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--share", action="store_true")
    p.add_argument("--preact", action="store_true")
    args = p.parse_args()
    main(args.mixup, args.alpha, args.batchsize, args.share, args.preact)
