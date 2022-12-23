# MaskingDepth
[[Project Page]](https://ku-cvlab.github.io/MaskingDepth/ "Project Page")
[[Paper]](https://arxiv.org/abs/2210.00939)

This code is the implementation of the paper <a href="https://arxiv.org/abs/2212.10806">Semi-Supervised Learning of Monocular Depth Estimation via Consistency Regularization with K-way Disjoint Masking</a> by Baek et al. 

Recently, successfully achieve in To gain insight from our exploration of the self-attention maps of diffusion models and for detailed explanations, .

## Environment
* [NGC pytorch 20.11-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (Docker ontainder)  
* additional require package (dotmap, wandb, einops, timm)
* NVIDIA RGX 3090

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

# Results

**Quantitative results on the KITTI dataset in a sparsely-supervised setting**










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
