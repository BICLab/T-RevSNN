# MindSpore-classification
Classification on CIFAR-10/100 and ImageNet with MindSpore.

## Install
* Install [MindSpore](https://www.mindspore.cn/)
* Clone recursively
  ```
  git clone https://github.com/Moranvl/MindSpore-classification.git
  ```
* Download CIFAR for C programs and place them in "./data/cifar-10" and "./data/cifar-100"
* Download [ImageNet 2012](https://image-net.org/challenges/LSVRC/index.php) and place them in "./data/ILSVRC2012/"

## Training
Please see the [Training recipes](TRAINING.md) for how to train the models.

## Results

### CIFAR
Top1 error rate on the CIFAR-10/100 benchmarks are reported. You may get different results when training your models with different random seed.
Note that the number of parameters are computed on the CIFAR-10 dataset.

| Model                     | Params (M) | CIFAR-10 (%) | CIFAR-100 (%) |
| -------------------       |------------|--------------|---------------|
| alexnet                   | 2.47       | 36.71        | 64.50         |
| vgg19_bn                  | 20.04      | 12.12        | 40.67         |
| ResNet-110                | 1.73       | 15.73        | 49.41         |
| PreResNet-110             | 1.73       | 13.34        | 43.96         |
| ResNeXt-29, 8x64          | 34.43      | 8.96         | 34.13         |
| ResNeXt-29, 16x64         | 68.16      | 9.03         | 34.46         |
| DenseNet-BC (L=100, k=12) | 0.77        | 10.3         | 31.81         |
| DenseNet-BC (L=190, k=40) | 25.62      | 11.99        | 29.84         |


### ImageNet
Single-crop (224x224) validation error rate is reported. 


| Model             | Params (M) | Top-1 Error (%) | Top-5 Error  (%) |
| -------------------  |------------|-----------------|------------------|
| ResNet            | 11.69      | 40.458          | 17.524           |

[//]: # (| ResNeXt-50  |       |                |                |)


## Supported Architectures

### CIFAR-10 / CIFAR-100
Since the size of images in CIFAR dataset is `32x32`, popular network structures for ImageNet need some modifications to adapt this input size. The modified models is in the package `models.cifar`:
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556) (Imported from [MindSpore-cifar](https://github.com/kuangliu/pytorch-cifar))
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] [Pre-act-ResNet](https://arxiv.org/abs/1603.05027)
- [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (Imported from [ResNeXt.MindSpore](https://github.com/prlz77/ResNeXt.pytorch))
- [x] [DenseNet](https://arxiv.org/abs/1608.06993)

[//]: # (- [x] [Wide Residual Networks]&#40;http://arxiv.org/abs/1605.07146&#41; &#40;Imported from [WideResNet-MindSpore]&#40;https://github.com/xternalz/WideResNet-pytorch&#41;&#41;)


### ImageNet
- [x] All models in `mindcv.models` (alexnet, vgg, resnet, densenet, inception_v3, squeezenet)

[//]: # (- [x] [ResNeXt]&#40;https://arxiv.org/abs/1611.05431&#41;)


## Contribute
Feel free to create a pull request if you find any bugs or you want to contribute (e.g., more datasets and more network structures).