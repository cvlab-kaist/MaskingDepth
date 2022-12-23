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
| Methods | # sup. frames | AbsRel ↓ | RMSE ↓ | δ↑ |
|---|---|---|---|---| 
| Baseline |  | 5.91 | 5.09 | 0.70 | 0.65 |
|  | V | 2.97 | 5.09 | 0.78 | 0.59 |
| V |  | 5.11 | 4.09 | 0.72 | 0.65 |
| V | V | 2.58 | 4.35 | 0.79 | 0.59 |


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
