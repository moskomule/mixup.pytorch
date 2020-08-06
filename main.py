from typing import Tuple

import hydra
import numpy as np
import torch
from homura import trainers, optim, lr_scheduler
from homura.metrics import accuracy
from homura.vision import DATASET_REGISTRY, MODEL_REGISTRY
from torch.nn import functional as F


def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)


def naive_cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor
                             ) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


class Trainer(trainers.SupervisedTrainer):
    def iteration(self,
                  data):
        input, target = data
        _target = target
        target = F.one_hot(target, self.num_classes)
        if self.is_train and self.cfg.mixup:
            if self.cfg.input_only:
                input = partial_mixup(input, np.random.beta(self.cfg.alpha + 1, self.cfg.alpha),
                                      torch.randperm(input.size(0), device=input.device, dtype=torch.long))
            else:
                input, target = mixup(input, target, np.random.beta(self.cfg.alpha, self.cfg.alpha))

        output = self.model(input)
        loss = self.loss_f(output, target)

        self.reporter.add('loss', loss.detach())
        self.reporter.add('accuracy', accuracy(output, _target))

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


@hydra.main("main.yaml")
def main(cfg):
    train_loader, test_loader, num_classes = DATASET_REGISTRY(cfg.data.name)(cfg.data.batch_size,
                                                                             return_num_classes=True,
                                                                             num_workers=4)
    model = MODEL_REGISTRY(cfg.model.name)(num_classes=num_classes)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=cfg.optim.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(200, 4, 5)

    with Trainer(model, optimizer, naive_cross_entropy_loss, scheduler=scheduler, cfg=cfg.model,
                 num_classes=num_classes) as trainer:
        for _ in trainer.epoch_range(200):
            trainer.train(train_loader)
            trainer.test(test_loader)
        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.4f}")


if __name__ == '__main__':
    main()
