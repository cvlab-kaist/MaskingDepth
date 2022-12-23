# MaskingDepth
[[Project Page]](https://ku-cvlab.github.io/MaskingDepth/ "Project Page")
[[Paper]](https://arxiv.org/abs/2210.00939)

This code is the implementation of the paper <a href="https://arxiv.org/abs/2212.10806">Semi-Supervised Learning of Monocular Depth Estimation via Consistency Regularization with K-way Disjoint Masking</a> by Baek et al. 

Recently, Semi-Supervised Leanring(SSL) strategy has accomplished successful achievements by leveraging unlabeled dataset. Inspired by success, we introduce consistency regularization approach which utilize unlabeled data.

for the first time we introduce consistency regularization framework for monocular depth estimation 

## Environment
* [NGC pytorch 20.11-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (Docker ontainder)  
* additional require packages(dotmap, wandb, einops, timm)
* NVIDIA RGX 3090s

In docker container
```
git clone https://github.com/KU-CVLAB/MaskingDepth.git
sh package_install.sh # install additionally package 
```

## Dataset
* [KITTI](https://www.cvlibs.net/datasets/kitti/)
* [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

## train
* supervised leanring:

edit conf/base_train.yaml file 
```
python trian.py
```

* semi-supervised leanring

edit conf/consistency_train.yaml file 
```
python consistency_train.py
```

# Evaluatopm
we refer 


# Results

**Quantitative results on the KITTI dataset in a sparsely-supervised setting**

full
| Methods | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ↑ |
|---|---|---|---|---|---|---| 
|Baseline | 0.076 +- 0.003 | 0.365 +- 0.004 | 3.290 +- 0.015 | 0.118 0.001 | 0.934 0.001 |
|Baseline+Self| 0.076 +- 0.002 | 0.367 +- 0.007 | 3.291 +- 0.020 | 0.117 0.001 | 0.933 0.002 |
|Ours+Self | 0.079 +- 0.001 | 0.379 +- 0.007 | 3.388 +- 0.019 | 0.121 0.009 | 0.929 0.001 |
|Ours | 0.074 +- 0.001 | 0.362 +- 0.001 | 3.253 +- 0.012 | 0.116 0.001 | 0.935 0.001 |

10000
| Methods |  AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ↑ |
|---|---|---|---|---|---|
|Baseline | 0.079 +- 0.001 | 0.379 +- 0.007 | 3.388 +- 0.019 | 0.121 0.009 | 0.929 0.001 |
|Baseline+Self| 0.078 +- 0.001 | 0.376 +- 0.006 | 3.347 +- 0.043 | 0.119 0.002 | 0.931 0.001 |
|Ours+Self | 0.076 +- 0.017 | 0.369 +- 0.004 | 3.311 +- 0.011 | 0.117 0.001 | 0.935 0.002 |
|Ours | 0.075 +- 0.002 | 0.362 +- 0.006 | 3.259 +- 0.020 | 0.116 0.001 | 0.934 0.003 |

1000
| Methods | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ↑ |
|---|---|---|---|---|---|
|Baseline | 0.098 +- 0.004 | 0.515 +- 0.030 | 3.785 +- 0.013 | 0.142 0.005 | 0.899 0.005 |
|Baseline+Self| 0.096 +- 0.002 | 0.523 +- 0.024 | 3.750 +- 0.033 | 0.140 0.002 | 0.900 0.004 |
|Ours+Self | 0.085 +- 0.017 | 0.430 +- 0.011 | 3.521 +- 0.012 | 0.129 0.012 | 0.918 0.010 |
|Ours | 0.088 +- 0.003 | 0.419 +- 0.007 | 3.490 +- 0.020 | 0.129 0.003 | 0.917 0.002 |

100
| Methods | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ↑ |
|---|---|---|---|---|---|
|Baseline | 0.135 +- 0.005 | 0.728 +- 0.019 | 4.585 +- 0.048 | 0.186 0.011 | 0.831 0.005 |
|Baseline+Self| 0.132 +- 0.004 | 0.759 +- 0.014 | 4.559 +- 0.044 | 0.184 0.003 | 0.834 0.004 |
|Ours+Self | 0.123 +- 0.003 | 0.747 +- 0.018 | 4.497 +- 0.042 | 0.181 0.005 | 0.839 0.005 |
|Ours | 0.128 +- 0.004 | 0.707 +- 0.013 | 4.295 +- 0.037 | 0.173 0.006 | 0.849 0.006 |

10
| Methods | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ↑ |
|---|---|---|---|---|---|
|Baseline | 0.201 +- 0.023 | 1.508 +- 0.045 | 6.163 +- 0.082 | 0.268 0.029 | 0.701 0.021 |
|Baseline+Self| 0.210 +- 0.020 | 1.322 +- 0.042 | 5.627 +- 0.080 | 0.265 0.027 | 0.711 0.016 |
|Ours+Self | 0.184 +- 0.011 | 1.265 +- 0.064 | 5.747 +- 0.080 | 0.243 0.007 | 0.727 0.018 |
|Ours | 0.197 +- 0.019 | 1.378 +- 0.032 | 5.650 +- 0.091 | 0.261 0.030 | 0.723 0.017 |


**Quantitative results on the KITTI dataset in a sparsely-supervised setting**
| SAG | CG | FID | sFID | Precision | Recall |
|---|---|---|---|---|---|
|  |  | 5.91 | 5.09 | 0.70 | 0.65 |
|  | V | 2.97 | 5.09 | 0.78 | 0.59 |
| V |  | 5.11 | 4.09 | 0.72 | 0.65 |
| V | V | 2.58 | 4.35 | 0.79 | 0.59 |


# Citation
Please consider citing our paper if you use this code. 
```
@article{baek2022semi,
  title={Semi-Supervised Learning of Monocular Depth Estimation via Consistency Regularization with K-way Disjoint Masking},
  author={Baek, Jongbeom and Kim, Gyeongnyeon and Park, Seonghoon and An, Honggyu and Poggi, Matteo and Kim, Seungryong},
  journal={arXiv preprint arXiv:2212.10806},
  year={2022}
}
```
