import os
from pkgutil import extend_path
import numpy as np
import PIL.Image as pil
import torch
import json

from .mono_dataset import MonoDataset

class Virtual_Kitti(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Virtual_Kitti, self).__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)

        self.rgb_head = 'rgb_'
        self.depth_head = 'depth_'
        self.depth_img_ext ='.png'

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip=False):
        color = self.loader(os.path.join(self.data_path, 'rgb', folder, 'frames/rgb/Camera_0',
                                 (self.rgb_head + str(frame_index).zfill(5) + self.img_ext)))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color

    def get_depth(self, folder, frame_index, side, do_flip=False):
        depth_path = os.path.join(self.data_path, 'depth', folder, 'frames/depth/Camera_0',
                                 (self.depth_head + str(frame_index).zfill(5) + self.depth_img_ext))
        depth_gt = pil.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32) * 80.0 / 65535.0
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt