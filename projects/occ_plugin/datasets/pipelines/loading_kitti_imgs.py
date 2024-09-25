# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import scipy

from mmdet.datasets.builder import PIPELINES

import os
import torch
from PIL import Image
from .loading_bevdet import PhotoMetricDistortionMultiViewImage, mmlabNormalize

from numpy import random

def depth_transform(depthmap, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    # convert depthmap to [u, v, d]
    valid_coords = np.nonzero(depthmap)
    valid_depth = depthmap[valid_coords[:, 0], valid_coords[:, 1]]
    cam_depth = np.concatenate((valid_coords[:, [1, 0]], 
                    valid_depth.reshape(-1, 1)), axis=1)
    
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False, colorjitter=False, 
                    img_norm_cfg=None, load_depth=False):

        self.is_train = is_train
        self.data_config = data_config
        self.load_depth = load_depth
        self.normalize_img = mmlabNormalize
        self.img_norm_cfg = img_norm_cfg
        
        self.colorjitter = colorjitter
        self.pipeline_colorjitter = PhotoMetricDistortionMultiViewImage()

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
    
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self, results, flip=None, scale=None):
                
        # load the monocular image for semantic kitti
        img_filenames = results['img_filename']   

        assert len(img_filenames) == 2
        
        img_filenames2 = img_filenames[1]

        img2 = mmcv.imread(img_filenames2, 'unchanged')
        img2 = Image.fromarray(img2)
        
        # perform image-view augmentation
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        img_augs2 = self.sample_augmentation(H=img2.height, 
                        W=img2.width, flip=flip, scale=scale)

        resize, resize_dims, crop, flip, rotate = img_augs2
        img2, post_rot2, post_tran2 = \
            self.img_transform(img2, post_rot, post_tran, resize=resize, 
                resize_dims=resize_dims, crop=crop,flip=flip, rotate=rotate)   

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        
        # intrins
        intrin2 = torch.Tensor(results['cam_intrinsic'][1])
        
        # extrins
        lidar2cam2 = torch.Tensor(results['lidar2cam'][1])
        cam2lidar2 = lidar2cam2.inverse()
        rot2 = cam2lidar2[:3, :3]
        tran2 = cam2lidar2[:3, 3]
        
        # output
        canvas2 = np.array(img2)
        
        if self.colorjitter and self.is_train:
            img2 = self.pipeline_colorjitter(img2)
        
        img2 = self.normalize_img(img2, img_norm_cfg=self.img_norm_cfg)
        
     
        if self.load_depth:
            depth_filename2 = img_filenames.replace('image_3', 'image_depth_annotated')
            # around 22% pixels have annotated depth
            depth2 = mmcv.imread(depth_filename2, 'unchanged')
            depth2 = torch.from_numpy(depth2.astype(np.float32)).float() / 256
            
            # TODO: do depth transform
            depth2 = depth_transform(depth2, resize, self.data_config['input_size'], 
                                    crop, flip, rotate)
        else:
            depth2 = torch.zeros(1)
        
        res2 = [img2, rot2, tran2, intrin2, post_rot, post_tran, depth2, cam2lidar2]
        res2 = [x[None] for x in res2]
        
        results['canvas2'] = canvas2

     
        img_filenames = img_filenames[0]

        img = mmcv.imread(img_filenames, 'unchanged')
        img = Image.fromarray(img)
        
        # perform image-view augmentation
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        # img_augs = self.sample_augmentation(H=img.height, 
        #                 W=img.width, flip=flip, scale=scale)
        img_augs = img_augs2

        resize, resize_dims, crop, flip, rotate = img_augs
        img, post_rot2, post_tran2 = \
            self.img_transform(img, post_rot, post_tran, resize=resize, 
                resize_dims=resize_dims, crop=crop,flip=flip, rotate=rotate)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        
        # intrins
        intrin = torch.Tensor(results['cam_intrinsic'][0])
        
        # extrins
        lidar2cam = torch.Tensor(results['lidar2cam'][0])
        cam2lidar = lidar2cam.inverse()
        rot = cam2lidar[:3, :3]
        tran = cam2lidar[:3, 3]
        
        # output
        canvas = np.array(img)
        
        if self.colorjitter and self.is_train:
            img = self.pipeline_colorjitter(img)
        
        img = self.normalize_img(img, img_norm_cfg=self.img_norm_cfg)
        
        if self.load_depth:
            depth_filename = img_filenames.replace('image_2', 'image_depth_annotated')
            # around 22% pixels have annotated depth
            depth = mmcv.imread(depth_filename, 'unchanged')
            depth = torch.from_numpy(depth.astype(np.float32)).float() / 256
            
            # TODO: do depth transform
            depth = depth_transform(depth, resize, self.data_config['input_size'], 
                                    crop, flip, rotate)
        else:
            depth = torch.zeros(1)
        
        res = [img, rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
        res = [x[None] for x in res]
        
        results['canvas'] = canvas


        return [res, res2 ]

    def __call__(self, results):
      
        results['img_inputs'] = self.get_inputs(results)
        
        return results