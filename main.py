import torch
from homura import trainers, reporters, optim, callbacks, lr_scheduler, Map
from homura.modules import to_onehot
from homura.vision.data import cifar10_loaders, cifar100_loaders
from homura.vision.models import resnet20, wrn28_2
from numpy.random import beta
from torch.nn import functional as F

MODELS = {"resnet20": resnet20,
          "wrn28_2": wrn28_2}

DATASETS = {"cifar10": cifar10_loaders,
            "cifar100": cifar100_loaders}

NUMCLASSES = {"cifar10": 10,
              "cifar100": 100}


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def naive_cross_entropy_loss(input, target):
    return - (input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


class MixupTrainer(trainers.SupervisedTrainer):
    def iteration(self, data):
        input, target = data
        if self.is_train:
            input, target = mixup(input, to_onehot(target, self.num_classes),
                                  beta(self.alpha, self.alpha))
            output = self.model(input)
            loss = self.loss_f(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            output = self.model(input)
            loss = F.cross_entropy(output, target)
        results = Map(loss=loss, output=output)
        return results


def main():
    Trainer = trainers.SupervisedTrainer if args.baseline else MixupTrainer
    model = MODELS[args.model](num_classes=NUMCLASSES[args.dataset])
    train_loader, test_loader = DATASETS[args.dataset](args.batch_size)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(args.steps, gamma=0.1)
    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]

    with reporters.TQDMReporter(range(args.epochs), callbacks=c) as tq, reporters.TensorboardReporter(c) as tb:
        trainer = Trainer(model, optimizer, F.cross_entropy if args.baseline else naive_cross_entropy_loss,
                          callbacks=[tq, tb],
                          scheduler=scheduler,
                          alpha=args.alpha,
                          num_classes=NUMCLASSES[args.dataset])
        for _ in tq:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs

    p = miniargs.ArgumentParser()
    p.add_int("--batch_size", default=128)
    p.add_int("--epochs", default=200)
    p.add_multi_int("--steps", default=[100, 150])
    p.add_str("--dataset", choices=list(DATASETS.keys()))
    p.add_str("--model", choices=list(MODELS.keys()))
    p.add_float("--alpha", default=1)
    p.add_true("--baseline")
    args = p.parse()

    main()
