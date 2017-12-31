import torch
import torch.nn.functional as F
from numpy import random
from tqdm import tqdm

from functions import variable, onehot, naive_cross_entropy_loss

cuda_available = torch.cuda.is_available()


def standard_train(model, optimizer, data_loader, data_length, loss_f):
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
        loss = loss_f(model(input), target)
        loss.backward()
        optimizer.step()
        loop_loss.append(loss.data[0] / data_length)
    print(f">>>(standard)loss: {sum(loop_loss):.2f}")


def mixup_train(model, optimizer, data_loaders, alpha, data_length, num_classes=10, share_lambda=False,
                loss_f=naive_cross_entropy_loss):
    """
    train function for mixup
    """
    model.train()
    loop_loss = []
    for (input1, target1), (input2, target2) in tqdm(zip(*data_loaders), total=len(data_loaders[0])):
        target1 = onehot(target1, num_classes)
        target2 = onehot(target2, num_classes)

        if share_lambda:
            # share a same lambda in each minibatch
            _lambda = random.beta(alpha, alpha)
        else:
            _lambda = torch.Tensor(random.beta(alpha, alpha, size=input1.size()[0]))

        input = _lambda * input1 + (1 - _lambda) * input2
        target = _lambda * target1 + (1 - _lambda) * target2

        input = variable(input)
        target = variable(target)
        optimizer.zero_grad()
        loss = loss_f(model(input), target)
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
    print(f">>>(test)accuracy: {sum(loop_accuracy):.2%}")
