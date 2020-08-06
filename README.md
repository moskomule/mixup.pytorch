# mixup.pytorch

An implementation of *mixup: Beyond Empirical Risk Minimization* by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz.

```
python main.py model.mixup={true,false} model.input_only={true,false}
```

mixup can (mathematically) avoid mixing labels by replacing `beta(alpha, alpha)` with `beta(alpha+1, alpha)`. `model.input_only=true` is to confirm this. (*)

## Requirements

* PyTorch==1.6.0
* torchvision==0.7.0
* homura `pip install -U git+https://github.com/moskoumule/homura@v2020.08`

## Results

### CIFAR10 on ResNet-20

|   ERM   |  mixup    |  mixup (`input_only=true`) |
|:--- |:--- |:--- |
|  0.923  | 0.932    |  0.931                      |

The results suggest that the alternative mixup strategy (*) is as effective as the original. 