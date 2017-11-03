# mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

Now I'm testing ResNet20 (proposed in He et al. 2015)+mixup with CIFAR10 dataset.

```
python exec.py [--mixup --alpha 1 --share]
```

runs ResNet. To enable mixup, specify `--mixup`, to change hyperparameter of beta distribution, use `--alpha FLOAT`. `--share` makes mixup to use the same factor `_lambda` in the same mini-batch.

## Requirements

* numpy
* PyTorch from GitHub's master branch (29, Oct)
* torchvision (0.19)
* tqdm

## Results

### CIFAR10

ResNet20 (proposed in He et al. 2015)+mixup(alpha=1) does not outperform ResNet20(EMP). Other hyperparamters are shared.
q
In the paper pre-act ResNet-18 is used, instead, so I'll test it later.

|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 91%            |
