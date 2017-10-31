# [WIP]mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

Now I'm testing mixup-ResNet(50) with CIFAR10 dataset.

```
python exec.py [--mixup --alpha 1]
```

runs ResNet. To enable mixup, specify `--mixup` and to change hyperparameter of beta distribution, use `--alpha FLOAT`.

## Requirements

* numpy
* PyTorch from GitHub's master branch (29, Oct)
* torchvision (0.19)
* tqdm

## Notice

Though it should work, I found that I'm using ResNet for ImageNet instead of CIFAR10. I'll implement CIFAR10 version so please wait the results.
