# mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

Now I'm testing ResNet20 (proposed in He et al. 2015)+mixup with CIFAR10 dataset.

```
python exec.py [--nomixup --alpha 1 --share --batchsize 64 --preact]
```

runs ResNet. To disable mixup, specify `--nomixup`, to change hyperparameter of beta distribution, use `--alpha FLOAT`. `--share` makes mixup to use the same factor `_lambda` in the same mini-batch.

## Requirements

* numpy
* PyTorch v0.3.0
* torchvision (0.19)
* tqdm

## Results

### CIFAR10

#### default ResNet

ResNet20 (proposed in He et al. 2015)+mixup(alpha=1) is equal to ResNet20 (EMP). Other hyperparamters are shared.

```
python exec.py [--nomixup --alpha 0.2]
```

As @hongyi-zhang, the author mentions in #1, $\alpha$ is recommended to be small such as 0.2 or 0.4 (otherwise use higher capacity networks).

|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 92%            |

#### preact ResNet

Preact ResNet20+mixup(alpha=1) a bit outperforms ResNet20(EMP).

```
python exec.py --preact [--mixup]
```

|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 92%            |
|-                 |  mainly 91 %   | mainly 92%     |