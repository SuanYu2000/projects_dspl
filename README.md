
# DSPL: Dynamically Scaled Hyperspherical Prototypes Learning for Open-Set Recognition




## Abstract

Traditional pattern recognition assumes a constant category set during training. In practical applications, unlabeled data often includes a large number of samples from unknown classes. Open-Set Recognition (OSR) addresses the challenge by constructing a realistic evaluation scenario, in which classifiers must correctly classify known classes and reliably recognize unknown class samples. This work proposes an open-set recognition model based on dynamically scaled hyperspherical prototypes, built upon the hyperspherical prototype network. The model employs a virtual-sample generation strategy to extend the hyperspherical prototype network, enabling class prototypes to be assigned to each known class and each potential unknown class. While sample recognition is performed using the angle similarity between samples and class prototypes, a new mechanism leveraging the norms of class prototypes is proposed, effectively improving discriminative accuracy. Extensive experiments on multiple benchmark datasets demonstrate that the proposed model achieves superior performance in both open-set and closed-set evaluations.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)
Hyperspherical Prototype Construction builds an extended candidate prototype set of size 
𝐾+𝑀, where virtual class samples are introduced via a Mixup strategy to model potential unknown categories. The Label-to-Prototype Assignment process is then employed to determine the optimal correspondence between class labels and candidate prototypes. For illustration, an example with four classes is shown in the figure, where the change in the ordering of color blocks reflects the update of the label–prototype matching relationship. Furthermore, the Adaptive Prototype Scaling Mechanism highlights the difference between the proposed approach and conventional hyperspherical methods that adopt fixed prototype magnitudes. In this stage, variations in the lengths of the color blocks indicate the adaptive adjustment of prototype magnitudes during training.
## Requirements
**Environments**

Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- yacs 0.1.8

## Usage

**Prototype Estimation**

One can generate equidistributed prototypes with desired dimension:

```bash
  python Prototype_Estimation.py --seed 100 --num_centroids 100 --batch_size 100 --space_dim 50 --num_epoch 1000
```

To train open set recognition models in paper, run this command:

```bash
python osr_DSPL.py --dataset <DATASET> --loss DSPLoss
```

To train closed set classifier models in paper, run this command:

```bash
python acc_DSPL.py --dataset <DATASET> --loss DSPLoss
```


## Results

**Performance Analysis of Open Set Recognition**

Comparison of AUROC and OSCR curves on CIFAR-100 and ImageNet-200 across training epochs.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


The AUROC Results of on Detecting Known and Unknown Samples.
| Method  | MNIST        | SVHN        | CIFAR10     | CIFAR+10    | CIFAR+50    | TinyImageNet |
|:--------|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| Softmax | 97.8 ± 0.2   | 88.6 ± 0.6  | 67.7 ± 3.2  | 81.6 ± 1.1  | 80.5 ± 1.0  | 57.7 ± 2.3   |
| OpenMax | 98.1 ± 0.2   | 89.4 ± 0.8  | 69.5 ± 3.2  | 81.7 ± 1.0  | 79.6 ± 0.8  | 57.6 ± 1.1   |
| HPN     | 99.1 ± 0.2   | 89.7 ± 1.6  | 78.8 ± 1.2  | 88.8 ± 2.4  | 85.6 ± 1.6  | 72.5 ± 2.0   |
| DL2PA   | 99.1 ± 0.1   | 89.2 ± 1.0  | 78.1 ± 0.8  | 88.4 ± 2.8  | 85.7 ± 1.6  | 72.5 ± 2.1   |
| RPL     | 98.9 ± 0.1   | 93.4 ± 0.5  | 82.7 ± 1.4  | 84.2 ± 1.0  | 83.2 ± 0.7  | 68.8 ± 1.4   |
| ARPL    | 99.6 ± 0.1   | 95.6 ± 0.5  | 89.4 ± 1.5  | 96.5 ± 0.5  | 93.9 ± 0.2  | 75.7 ± 1.1   |
| **Ours**| 99.5 ± 0.1 | **95.9 ± 0.3** | **90.8 ± 1.0** | **96.9 ± 0.3** | **94.7 ± 0.4** | **78.7 ± 1.4** |


The Open Set Classification Rate (OSCR) Curve Results of Open Set Recognition
| Method  | MNIST        | SVHN        | CIFAR10     | CIFAR+10    | CIFAR+50    | TinyImageNet |
|:--------|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| Softmax | 99.2 ± 0.1   | 92.8 ± 0.4  | 83.8 ± 1.5  | 90.9 ± 1.3  | 88.5 ± 0.7  | 60.8 ± 5.1   |
| HPN     | 99.0 ± 1.2   | 88.0 ± 1.7  | 75.8 ± 1.9  | 87.0 ± 2.5  | 83.9 ± 1.7  | 58.6 ± 5.6   |
| DL2PA   | 92.4 ± 1.8   | 84.5 ± 1.6  | 76.2 ± 1.7  | 86.6 ± 2.9  | 84.1 ± 1.7  | 58.4 ± 5.6   |
| RPL     | 98.9 ± 0.1   | 93.4 ± 0.5  | 82.7 ± 1.4  | 84.2 ± 1.0  | 83.2 ± 0.7  | 68.8 ± 1.4   |
| ARPL    | 99.4 ± 0.1   | 93.5 ± 0.5  | 86.1 ± 1.9  | 93.8 ± 1.0  | 91.4 ± 0.6  | 61.6 ± 3.0   |
| **Ours**| 99.3 ± 0.1 | **93.7 ± 0.3** | **87.9 ± 1.1** | **94.5 ± 0.6** | **92.5 ± 0.2** | **67.2 ± 4.1** |

**Closed Set Classification Performance**

Comparison on CIFAR-100 with different embedding dimensions d.
| Method          | d = 10 | d = 25 | d = 50 | d = 100 |
|:----------------|:------:|:------:|:------:|:-------:|
| PSC             | 25.67  | 60.00  | 60.60  | 62.10   |
| Word2Vec [40]   | 29.00  | 44.50  | 54.30  | 57.60   |
| HPN [39]        | 50.91  | 63.00  | 65.06  | 63.41   |
| DL2PA           | 57.83  | 63.05  | 66.14  | 65.31   |
| **Ours**        | **58.75** | **64.91** | **68.31** | **69.66** |

Comparison on ImageNet-200 with different embedding dimensions d.
| Method          | d = 10 | d = 25 | d = 50 | d = 100 |
|:----------------|:------:|:------:|:------:|:-------:|
| PSC             | 25.67  | 60.00  | 60.60  | 62.10   |
| Word2Vec [40]   | 29.00  | 44.50  | 54.30  | 57.60   |
| HPN [39]        | 50.91  | 63.00  | 65.06  | 63.41   |
| DL2PA           | 57.83  | 63.05  | 66.14  | 65.31   |
| **Ours**        | **58.75** | **64.91** | **68.31** | **69.66** |

## Contact
If there is a question regarding any part of the code, or it needs further clarification, please create an issue or send me an email:
[haozechao@sxu.edu.cn](mailto:me00018@mix.wvu.edu).
