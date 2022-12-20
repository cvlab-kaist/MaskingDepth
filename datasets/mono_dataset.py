# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from math import degrees

import os
import random
from tabnanny import check
# from BLP_depth.loss import train_mode
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.nn.functional as F

import torch.utils.data as data
from torchvision import transforms
import torchvision.utils
from  .autoaugment import rand_augment_transform, Cutout, _rotate_level_to_arg

ROLL    = 0
ROTATE  = 1
CUTOUT  = 2

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 use_box,
                 gt_num = -1,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS
        self.use_box = use_box
        self.is_train = is_train
        self.img_ext = img_ext
        self.MAX_BOX_NUM = 8
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.load_depth = self.check_depth()
        self.rand_aug = rand_augment_transform(config_str = 'rand-n{}-m{}-mstd0.5'.format(3, 5), hparams={})
        

        # full_data_num = len(self.filenames))
        self.gt_use = set([i for i in range(len(self.filenames))])
        if not(gt_num == -1):
            use_flag = [i for i in range(len(self.filenames))]
            random.shuffle(use_flag)
            self.gt_use = self.gt_use - set(use_flag[:(len(self.filenames) - gt_num)])
            

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        origin = inputs["color"]
        do_weak = self.is_train and random.random() > 0.5
        Geo_aug = random.randint(0,3)
        
        #weak aug
        if do_weak:
            weak_aug = transforms.ColorJitter(
                            self.brightness, self.contrast, self.saturation, self.hue)
            inputs["color"] = self.to_tensor(self.resize(weak_aug(inputs["color"])))
            torchvision.utils.save_image(inputs["color"], "./aaa.png")
        else:
            inputs["color"] = self.to_tensor(self.resize(inputs["color"]))
                        
        #strong aug
        inputs["color_aug"] = self.to_tensor(self.resize(self.rand_aug(origin)))        
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
 
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,

            ("color")                               for raw colour images,
            ("color_aug")                           for augmented colour images,
            ("K") or ("inv_K")                      for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
            "box"                                   for using bounding box
        """
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5

        if type(self).__name__ in "CityscapeDataset":
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            inputs.update(self.get_color(folder, frame_index, side, do_flip))

        elif type(self).__name__ in "NYUDataset" or type(self).__name__ in "Virtual_Kitti":
            if type(self).__name__ in "NYUDataset":
                split_com = '/'
            else:
                split_com = ' '
           
            folder = os.path.join(*self.filenames[index].split(split_com)[:-1])
            
            if self.is_train:
                frame_index = int(self.filenames[index].split(split_com)[-1])
                side = None
                inputs["color"] = self.get_color(folder, frame_index, side, do_flip)
            else:
                frame_index = self.filenames[index].split(split_com)[-1]
                side= None
                inputs["color"] = self.get_color(folder, frame_index, side, do_flip)

        else:
            line = self.filenames[index].split()
            folder = line[0]
            frame_index = int(line[1])
            side = line[2]
            inputs["color"] = self.get_color(folder, frame_index, side, do_flip)

            K = self.K.copy()
            K[0, :] *= self.width 
            K[1, :] *= self.height

            inv_K = np.linalg.pinv(K)

            inputs["K"] = torch.from_numpy(K)
            inputs["inv_K"] = torch.from_numpy(inv_K)

            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        self.preprocess(inputs)
        
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError