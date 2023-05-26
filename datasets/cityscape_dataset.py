import os
import numpy as np
import PIL.Image as pil
import torch
import json

from .mono_dataset import MonoDataset

class CityscapeDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    RAW_WIDTH = 2048
    RAW_HEIGHT = 768

    def __init__(self, *args, **kwargs):
        super(CityscapeDataset, self).__init__(*args, **kwargs)
        self.ratio_h = 0

        # if self.train_mode == "sup":
        #     raise ValueError("Cityscapes datase dosen't have Groud truth depth map. please change setting(train_mode = unsup)")

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side

    def check_depth(self):
        return True
    
    def load_intrinsics(self, city, frame_name):
        split = "train"
        camera_file = os.path.join(self.data_path, 'camera_trainvaltest', 'camera',
                                split, city, city + '.json')
        with open(camera_file, 'r') as f:
            camera = json.load(f)
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        intrinsics = np.array([[fx, 0, u0, 0],
                            [0, fy, v0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).astype(np.float32)
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics

    def get_color(self, city, frame_name, side, do_flip=False):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        # color_1 = self.loader(self.get_image_path(city, self.adjacent_frame_path(frame_name, -1)))
        color0 = self.loader(self.get_image_path(city, frame_name))
        # color1 = self.loader(self.get_image_path(city, self.adjacent_frame_path(frame_name, 1)))

        # color_1 = np.array(color_1)
        color0 = np.array(color0)
        # color1 = np.array(color1)

        h = int(color0.shape[0] * 3 / 4)
        
        # inputs[("color", -1, -1)] = pil.fromarray(color_1[:h,:])
        color  = pil.fromarray(color0[ :h, :])
        # inputs[("color", 1, -1)]  = pil.fromarray(color1[ :h, :])

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, city, frame_name):
        split = "train"
        return os.path.join(self.data_path, split, city, "{}_leftImg8bit.jpg".format(frame_name))

    def adjacent_frame_path(self, frame_name, rel_pos):
        path_split = frame_name.split('_')
        path_split[-1] = str(int(path_split[-1]) + rel_pos).zfill(6)
        adj_frame_path = '_'.join(path_split)
        return adj_frame_path
        
    def get_depth(self, folder, frame_index, side, do_flip = False):
        depth_path = os.path.join(
            self.data_path,
            "disparity_sequence/train",
            folder,
            "{}_disparity.png".format(frame_index))
        
        depth_gt = pil.open(depth_path)
        depth_gt = np.array(depth_gt).astype(np.float32) / 32257.0
        h = int(depth_gt.shape[0] * 3 / 4)
        depth_gt = depth_gt[self.ratio_h:h,:]
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    # BJB EDIT
    def get_Bbox(self, folder, frame_index, side, do_flip = False):
        #file path        
        box_path = os.path.join(self.data_path.replace('leftImg8bit_sequence', 'gtBboxCityPersons', 1), 
                    'train', folder, "{}_gtBboxCityPersons.json".format(frame_index))
        
        if not (os.path.isfile(box_path)):
            return -torch.ones(self.MAX_BOX_NUM, 5)

        #read file
        with open(box_path) as f:
            json_data = json.load(f)
            h = json_data['imgHeight']
            w = json_data['imgWidth']

            boxes = []
            
            for inst_info in json_data['objects']:
                if inst_info['label'] == 'pedestrian':
                    box = [1.0]
                elif inst_info['label'] == 'sitting person':
                    box = [2.0]
                else:
                    box = [3.0]
                coord = inst_info['bbox']       # x,y(left top), w, h

                # (*4/3) parameter consider croppoing along height axis !!!
                box.append(float(coord[0] / w))                         # x1 left
                box.append(float(coord[1] / h * 4 / 3))                 # y1 top
                box.append(float((coord[0] + coord[2]) / w))            # x2 right
                box.append(float((coord[1] + coord[3]) / h  * 4 / 3))   # y1 bottom
                
                if do_flip:#box flip
                    box[1] = round(1 - box[1],4)
                    box[3] = round(1 - box[3],4)

                    boxes.append(box)   

                else:
                    boxes.append(box)   

        # Fixed Batch Size
        if len(boxes) >= self.MAX_BOX_NUM:
            return torch.tensor(boxes[:self.MAX_BOX_NUM])
        else:        
            return torch.cat((torch.tensor(boxes), -torch.ones((self.MAX_BOX_NUM - len(boxes)), 5)), dim=0)
