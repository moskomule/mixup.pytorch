# mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

```
python main.py [--batch_size 128 --epochs 400 --steps 250 300 --dataset cifar10 --model resnet20 --alpha 1 [--baseline]]
```

## Requirements

* PyTorch v1.0
* torchvision (v0.2.2)
* homura `pip install -U git+https://github.com/moskoumule/homura` (*Use dev!*)

## Results

### CIFAR10 on ResNet-20

|                  | EMP            | mixup          |
|:-------------    | :------------- | :------------- |
|max. test accuracy|  92%           | 93%            |