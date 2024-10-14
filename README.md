# High-Performance Temporal Reversible Spiking Neural Networks with $O(L)$ Training Memory and $O(1)$ Inference Cost ([ICML2024 spotlight](https://proceedings.mlr.press/v235/hu24q.html))

Peking University; BICLab, Institute of Automation, Chinese Academy of Sciences

TODO:

- [x] Upload train and test scripts.
- [ ] Upload checkpoints.

## Abstract

Multi-timestep simulation of brain-inspired Spiking Neural Networks (SNNs) boost memory requirements during training and increase inference energy cost. Current training methods cannot simultaneously solve both training and inference dilemmas. This work proposes a novel Temporal Reversible architecture for SNNs (T-RevSNN) to jointly address the training and inference challenges by altering the forward propagation of SNNs. We turn off the temporal dynamics of most spiking neurons and design multi-level temporal reversible interactions at temporal turn-on spiking neurons, resulting in a $\mathcal{O}(L)$ training memory. Combined with the temporal reversible nature, we redesign the input encoding and network organization of SNNs to achieve $\mathcal{O}(1)$ inference energy cost. Then, we finely adjust the internal units and residual connections of the basic SNN block to ensure the effectiveness of sparse temporal information interaction. T-RevSNN achieves excellent accuracy on ImageNet, while the memory efficiency, training time acceleration and inference energy efficiency can be significantly improved by $8.6 \times$, $2.0 \times$ and $1.6 \times$, respectively. This work is expected to break the technical bottleneck of significantly increasing memory cost and training time for large-scale SNNs while maintaining both high performance and low inference energy cost.

## Classification

### Results on Imagenet-1K

### Train

The hyper-parameters are in `./confings/`.

Train:

```shell
sh run.sh
```

### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```
@InProceedings{t_revsnn,
  title = 	 {High-Performance Temporal Reversible Spiking Neural Networks with $\mathcal{O}(L)$ Training Memory and $\mathcal{O}(1)$ Inference Cost},
  author =       {Hu, Jiakui and Yao, Man and Qiu, Xuerui and Chou, Yuhong and Cai, Yuxuan and Qiao, Ning and Tian, Yonghong and Xu, Bo and Li, Guoqi},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {19516--19530},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/hu24q/hu24q.pdf},
  url = 	 {https://proceedings.mlr.press/v235/hu24q.html}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `jkhu29@stu.pku.edu.cn` and `manyao@ia.ac.cn`.

## Thanks

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[revcol](https://github.com/megvii-research/RevCol/)