import torch
import torch.nn.functional as F
from numpy import random
from tqdm import tqdm

from functions import variable, onehot, naive_cross_entropy_loss

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
