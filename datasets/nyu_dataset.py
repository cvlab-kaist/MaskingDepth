import os

import cv2
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
import json

from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        self.RAW_WIDTH = 640
        self.RAW_HEIGHT = 480
        # self.K = np.array([[518.8579,  0, 325.5824, 0],
        #                    [0,  519.4696, 253.7361, 0],
        #                    [0,         0,        1, 0],
        #                    [0,         0,        0, 1]], dtype=np.float32)

        # self.K[0, :] /= self.RAW_WIDTH 
        # self.K[1, :] /= self.RAW_HEIGHT

        # if self.train_mode == "unsup":
        #     raise ValueError("NYUDataset dosen't deal with reconstruction loss. please change setting(train_mode = sup)")
        self.crop = transforms.CenterCrop((self.height - 20 , self.width - 20))


    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip=False):
        if side is not None:
            raise ValueError("NYUDataset doesn't know how to deal with sides")
            
        if self.is_train:
            color = self.loader(os.path.join(self.data_path, folder, (str(frame_index) + self.img_ext)))
        else:
            color = self.loader(os.path.join(self.data_path, folder, frame_index))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        color = self.crop(color)
        
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        if self.is_train:
            depth_path = os.path.join(self.data_path, folder, (str(frame_index) + '.png'))
            depth_gt = pil.open(depth_path)
            depth_gt = np.array(depth_gt).astype(np.float32) / 25.50
        else:
            depth_path = os.path.join(self.data_path, folder, frame_index.replace('colors','depth'))
            depth_gt = pil.open(depth_path)
            depth_gt = np.array(depth_gt).astype(np.float32) / 1000.0
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        
        #crop
        depth_gt = cv2.resize(depth_gt[10:-10, 10:-10], (self.RAW_WIDTH, self.RAW_HEIGHT))
        return depth_gt
