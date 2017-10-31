import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy import random
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from tqdm import tqdm

cuda_available = torch.cuda.is_available()


def variable(t):
    if cuda_available:
        t = t.cuda()
    return Variable(t)


def onehot(t, num_classes):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    assert isinstance(t, torch.LongTensor)
    return torch.zeros(t.size()[0], num_classes).scatter_(1, t.view(-1, 1), 1)


# def he_init(layer):
#     if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
#         nn.init.kaiming_normal(layer.weight)


def naive_cross_entropy_loss(input, target, size_average=True):
    """
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)

    input = torch.log(input.clamp(1e-5, 1))
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def standard_train(model, optimizer, data_loader, data_length):
    """
    train function for mixup
    """
    model.train()
    loop_loss = []
    for (input, target) in tqdm(data_loader):
        if cuda_available:
            input = variable(input)
            target = variable(onehot(target, 10))
        optimizer.zero_grad()
        loss = naive_cross_entropy_loss(F.softmax(model(input), dim=1), target)
        loss.backward()
        optimizer.step()
        loop_loss.append(loss.data[0] / data_length)
    print(f">>>(standard)loss: {sum(loop_loss):.2f}")


def mixup_train(model, optimizer, data_loaders, alpha, data_length, num_classes=10):
    """
    train function for mixup
    """
    model.train()
    loop_loss = []
    for (input1, target1), (input2, target2) in tqdm(zip(*data_loaders), total=len(data_loaders[0])):
        _lambda = torch.Tensor(random.beta(alpha, alpha, size=input1.size()[0]))

        target1 = onehot(target1, num_classes)
        target2 = onehot(target2, num_classes)

        _input_lambda = _lambda.view(-1, 1, 1, 1)
        _target_lambda = _lambda.view(-1, 1)
        input = _input_lambda * input1 + (1 - _input_lambda) * input2
        target = _target_lambda * target1 + (1 - _target_lambda) * target2

        input = variable(input)
        target = variable(target)
        optimizer.zero_grad()
        loss = naive_cross_entropy_loss(F.softmax(model(input), dim=1), target)
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


def main(use_mixup, alpha, batch_size):
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
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train),
            batch_size=batch_size,
            shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, transform=transform_train), batch_size=batch_size,
            shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform_test), batch_size=batch_size,
            shuffle=True)
    resnet = models.resnet50(num_classes=10)
    resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    # resnet.apply(he_init)
    if cuda_available:
        resnet.cuda()
    optimizer = optim.SGD(params=resnet.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for idx in range(200):
        print(f"epoch: {idx}")
        scheduler.step()
        if use_mixup:
            mixup_train(model=resnet, optimizer=optimizer,
                        data_loaders=(train_loader_1, train_loader_2), alpha=alpha,
                        data_length=len(train_loader_1))
        else:
            standard_train(model=resnet, optimizer=optimizer,
                           data_loader=train_loader_1, data_length=len(train_loader_1))
        test(model=resnet, data_loader=test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--batchsize", type=int, default=64)
    args = p.parse_args()
    main(args.mixup, args.alpha, args.batchsize)
