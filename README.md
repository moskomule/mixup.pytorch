# mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

Now I'm testing ResNet20 (proposed in He et al. 2015)+mixup with CIFAR10 dataset.

```
python exec.py [--mixup --alpha 1 --share --batchsize 64 --preact]
```

runs ResNet. To enable mixup, specify `--mixup`, to change hyperparameter of beta distribution, use `--alpha FLOAT`. `--share` makes mixup to use the same factor `_lambda` in the same mini-batch.

## Requirements

* numpy
* PyTorch from GitHub's master branch (29, Oct)
* torchvision (0.19)
* tqdm

## Results

### CIFAR10

#### default ResNet

ResNet20 (proposed in He et al. 2015)+mixup(alpha=1) does not outperform ResNet20(EMP). Other hyperparamters are shared.

```
python exec.py [--mixup]
```

|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 91%            |

#### preact ResNet

Preact ResNet20+mixup(alpha=1) a bit outperforms ResNet20(EMP).

```
python exec.py --preact [--mixup]
```


|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 92%            |
|-                 |  mainly 91 %   | mainly 92%     |