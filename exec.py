import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from cifar import resnet20, preact_resnet20
from utils import naive_cross_entropy_loss, mixup_train, standard_train, test

cuda_available = torch.cuda.is_available()

losses = {"naive": naive_cross_entropy_loss,
          "l1": F.l1_loss,
          "mse": F.mse_loss}


def get_loader(is_mixup, batch_size):
    save_dir = "~/.torch/data/cifar10"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(save_dir, train=False, transform=transform_test), batch_size=batch_size,
            shuffle=True)

    if is_mixup:

        train_loader_1 = torch.utils.data.DataLoader(
                datasets.CIFAR10(save_dir, train=True, download=True, transform=transform),
                batch_size=batch_size,
                shuffle=True)
        train_loader_2 = torch.utils.data.DataLoader(
                datasets.CIFAR10(save_dir, train=True, transform=transform), batch_size=batch_size,
                shuffle=True)
        train_loader = (train_loader_1, train_loader_2)

    else:

        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(save_dir, train=True, download=True, transform=transform),
                batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def main(nomixup, alpha, batch_size, share, preact, lossf, lr):
    train_loader, test_loader = get_loader(not nomixup, batch_size)

    model = preact_resnet20() if preact else resnet20()
    if cuda_available:
        model.cuda()

    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    for idx in range(200):
        print(f"epoch: {idx}")
        scheduler.step()
        if not nomixup:
            mixup_train(model=model, optimizer=optimizer,
                        data_loaders=train_loader, alpha=alpha,
                        data_length=len(train_loader[0]), share_lambda=share, loss_f=losses[lossf])
        else:
            standard_train(model=model, optimizer=optimizer,
                           data_loader=train_loader, data_length=len(train_loader), loss_f=losses[lossf])
        test(model=model, data_loader=test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--nomixup", action="store_true")
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--share", action="store_true")
    p.add_argument("--preact", action="store_true")
    p.add_argument("--lossf", default="naive")
    p.add_argument("--lr", type=float, default=1e-1)
    args = p.parse_args()
    main(**vars(args))
