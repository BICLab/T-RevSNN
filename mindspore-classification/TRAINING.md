
## CIFAR-10

#### AlexNet
```
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet --gpu-id 0
```


#### VGG19 (BN)
```
python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn --gpu-id 0
```

#### ResNet-110
```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 --gpu-id 2 --train-batch 64 --test-batch 64 --workers 4 
```

#### ResNet-1202
```
python cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202  --gpu-id 2
```

#### PreResNet-110
```
python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110  --gpu-id 2 --train-batch 128
```

#### ResNeXt-29, 8x64d
```
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d --gpu-id 2 --train-batch 32
```
#### ResNeXt-29, 16x64d
```
python cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d  --gpu-id 2 --train-batch 32 --workers 8 --test-batch 32; /usr/bin/shutdown
```

#### DenseNet-BC (L=100, k=12)
**Note**: 
* DenseNet use weight decay value `1e-4`. Larger weight decay (`5e-4`) if harmful for the accuracy (95.46 vs. 94.05) 
* Official batch size is 64. But there is no big difference using batchsize 64 or 128 (95.46 vs 95.11).

```
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12 --gpu-id 2 --train-batch 16 --test-batch 16
```

#### DenseNet-BC (L=190, k=40) 
```
python cifar.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L190-k40 --gpu-id 2 --train-batch 8 --test-batch 8 --workers 8 --resume checkpoints/cifar10/densenet-bc-L190-k40/checkpoint.ckpt --start-epoch 52
```

## CIFAR-100

#### AlexNet
```
python cifar.py -a alexnet --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1  --gpu-id -2 --train-batch 256
```

#### VGG19 (BN)
```
python cifar.py -a vgg19_bn --dataset cifar100 --checkpoint checkpoints/cifar100/vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1  --gpu-id -2 --train-batch 256
```

#### ResNet-110
```
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 --gpu-id -2 --train-batch 32 --test-batch 32 
```

#### ResNet-1202
```
python cifar.py -a resnet --dataset cifar100 --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-1202  --gpu-id 0
```

#### PreResNet-110
```
python cifar.py -a preresnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/preresnet-110  --gpu-id -2 --train-batch 128 --test-batch 128
```

#### ResNeXt-29, 8x64d
```
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 8 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-8x64d --schedule 150 225 --wd 5e-4 --gamma 0.1 --gpu-id 2 --train-batch 128 --test-batch 128 --workers 8; /usr/bin/shutdown
```
#### ResNeXt-29, 16x64d
```
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 16 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-16x64d --schedule 150 225 --wd 5e-4 --gamma 0.1 --gpu-id 2 --train-batch 128 --test-batch 128 --workers 16; /usr/bin/shutdown
```

#### DenseNet-BC (L=100, k=12)
```
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12 --gpu-id 2 --train-batch 128 --test-batch 128 --workers 8; /usr/bin/shutdown
```

#### DenseNet-BC (L=190, k=40) 
```
python cifar.py -a densenet --dataset cifar100 --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-L190-k40 --gpu-id 2 --train-batch 16 --test-batch 16 --workers 8; /usr/bin/shutdown
```

## ImageNet
### ResNet-18
```
python imagenet.py -a resnet --data ./data/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18 --gpu-id 2 --train-batch 64 --test-batch 64 --workers 4
python imagenet.py -a resnet --data ./data/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18 --gpu-id 2 --train-batch 64 --test-batch 64 --workers 4 --resume checkpoints/imagenet/resnet18/checkpoint.ckpt --start-epoch 52 --best-acc 57.878 --lr 0.01
```

### ResNeXt-50 (32x4d)
*(Originally trained on 8xGPUs)*
```
python imagenet.py -a resnext50 --base-width 4 --cardinality 32 --data ./data/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnext50-32x4d --gpu-id 2 --train-batch 64 --test-batch 64 --workers 8
```