import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy import random
from torchvision import datasets, transforms
from tqdm import tqdm

from functions import variable, onehot, naive_cross_entropy_loss
from cifar import resnet20

cuda_available = torch.cuda.is_available()


def standard_train(model, optimizer, data_loader, data_length):
    """
    train in standard way
    """
    model.train()
    loop_loss = []
    for (input, target) in tqdm(data_loader):
        if cuda_available:
            input = variable(input)
            target = variable(onehot(target, 10))
        optimizer.zero_grad()
        loss = naive_cross_entropy_loss(model(input), target)
        loss.backward()
        optimizer.step()
        loop_loss.append(loss.data[0] / data_length)
    print(f">>>(standard)loss: {sum(loop_loss):.2f}")


def mixup_train(model, optimizer, data_loaders, alpha, data_length, num_classes=10, share_lambda=False):
    """
    train function for mixup
    """
    model.train()
    loop_loss = []
    normalize_input = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    for (input1, target1), (input2, target2) in tqdm(zip(*data_loaders), total=len(data_loaders[0])):
        target1 = onehot(target1, num_classes)
        target2 = onehot(target2, num_classes)

        if share_lambda:
            # share a same lambda in each minibatch
            _lambda = random.beta(alpha, alpha)
            _input_lambda, _target_lambda = _lambda, _lambda
        else:
            _lambda = torch.Tensor(random.beta(alpha, alpha, size=input1.size()[0]))
            _input_lambda = _lambda.view(-1, 1, 1, 1)
            _target_lambda = _lambda.view(-1, 1)

        input = _input_lambda * input1 + (1 - _input_lambda) * input2
        # normalize after mixup
        input = normalize_input(input)
        target = _target_lambda * target1 + (1 - _target_lambda) * target2

        input = variable(input)
        target = variable(target)
        optimizer.zero_grad()
        loss = naive_cross_entropy_loss(model(input), target)
        loss.backward()
        optimizer.step()
        loop_loss.append(loss.data[0] / data_length)
    print(f">>>(mixup)loss: {sum(loop_loss):.2f}")


def test(model, data_loader):
    model.eval()
    loop_accuracy = []
    for (input, target) in data_loader:
        if cuda_available:
            input = variable(input)
            target = variable(target)
        output = F.softmax(model(input), dim=1)
        loop_accuracy.append((output.data.max(1)[1] == target.data).sum() / len(data_loader.dataset))
    print(f">>>(test)accuracy: {sum(loop_accuracy):.2f}")


def main(use_mixup, alpha, batch_size, share):
    transform_for_mixup = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
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
    resnet = resnet20()
    if cuda_available:
        resnet.cuda()
    optimizer = optim.SGD(params=resnet.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
    for idx in range(300):
        print(f"epoch: {idx}")
        scheduler.step()
        if use_mixup:
            mixup_train(model=resnet, optimizer=optimizer,
                        data_loaders=(train_loader_1, train_loader_2), alpha=alpha,
                        data_length=len(train_loader_1), share_lambda=share)
        else:
            standard_train(model=resnet, optimizer=optimizer,
                           data_loader=train_loader, data_length=len(train_loader))
        test(model=resnet, data_loader=test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    main(args.mixup, args.alpha, args.batchsize, args.share)
