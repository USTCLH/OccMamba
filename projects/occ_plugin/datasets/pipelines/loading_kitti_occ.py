# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.datasets.builder import PIPELINES

import os
import yaml
import torch
from PIL import Image

from numpy import random

def bev_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, transform_center=None):
    # for semantic_kitti, the transform origin is not zero, but the center of the point cloud range
    assert transform_center is not None
    trans_norm = torch.eye(4)
    trans_norm[:3, -1] = - transform_center
    trans_denorm = torch.eye(4)
    trans_denorm[:3, -1] = transform_center
    
    # bird-eye-view rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    # I @ flip_x @ flip_y
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    # denorm @ flip_x @ flip_y @ rotation @ normalize
    bda_mat = trans_denorm @ flip_mat @ rot_mat @ trans_norm
    
    # apply transformation to the 3D volume, which is tensor of shape [X, Y, Z]
    voxel_labels = voxel_labels.numpy().astype(np.uint8)
    if not np.isclose(rotate_degree, 0):
        scipy.ndimage.interpolation.rotate(voxel_labels, rotate_degree, output=voxel_labels,
                mode='constant', order=0, cval=255, axes=(0, 1), reshape=False)
    
    if flip_dy:
        voxel_labels = voxel_labels[:, ::-1]
    
    if flip_dx:
        voxel_labels = voxel_labels[::-1]
    
    voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat

@PIPELINES.register_module()
class LoadSemKittiAnnotation():
    def __init__(self, bda_aug_conf, is_train=True, apply_bda=False, test_bda=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                cls_metas='semantic-kitti.yaml',):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.test_bda = test_bda
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2
        self.apply_bda = apply_bda

        with open(cls_metas, 'r') as stream:
            nusc_cls_metas = yaml.safe_load(stream)
            self.learning_map = nusc_cls_metas['learning_map']

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""

        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def __call__(self, results):
        if type(results['gt_occ']) is list:
            gt_occ = [torch.tensor(x) for x in results['gt_occ']]
        elif results['gt_occ'] is not None:
            gt_occ = torch.tensor(results['gt_occ'])   
        else:
            gt_occ = torch.zeros((256, 256, 32))

        # load semactic labels
        points = np.fromfile(results['pts_filename'], dtype=np.float32, count=-1).reshape(-1, 4)[..., :3]
        label_path = results['pts_filename'].replace("velodyne", "labels").replace(".bin", ".label")
        annotated_data = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
        
        sem_labels = annotated_data & 0xFFFF
        inst_labels = annotated_data.astype(np.float32)

        sem_labels = (np.vectorize(self.learning_map.__getitem__)(sem_labels)).astype(np.float32)
        idx = np.arange(points.shape[0])
        sem_labels = sem_labels[idx]
        lidarseg = np.concatenate([points, sem_labels[:, None]], axis=-1)
        results['points_occ'] = torch.from_numpy(lidarseg).float()

        # bda
        results['origin_points'] = results['points']
        if self.apply_bda and (self.is_train or self.test_bda):
            rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
            gt_occ, bda_rot = bev_transform(gt_occ, rotate_bda, scale_bda, flip_dx, flip_dy, self.transform_center)

            # perform bird-eye-view augmentation for lidar_points
            lidar_points = results['points'].tensor[:, :3]
            if bda_rot.shape[-1] == 4:
                homo_lidar_points = torch.cat((lidar_points, torch.ones(lidar_points.shape[0], 1)), dim=1)
                homo_lidar_points = homo_lidar_points @ bda_rot.t()
                lidar_points = homo_lidar_points[:, :3]
            else:
                lidar_points = lidar_points @ bda_rot.t()

            results['points'].tensor[:, :3] = lidar_points
            results['points_occ'][:, :3] = lidar_points
        else:
            bda_rot = torch.eye(4)

        results['gt_occ'] = gt_occ

        # print(results['sequence'], results['frame_id'])
        # np.save('other/openocc_aug_points.npy', results['points'].tensor.data.cpu().numpy())
        # np.save('other/openocc_aug_gt.npy', results['gt_occ'].data.cpu().numpy())

        if 'img_inputs' in results.keys():
            imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors= results['img_inputs'][0]
            imgs2, rots2, trans2, intrins2, post_rots2, post_trans2, gt_depths2, sensor2sensors2 = results['img_inputs'][1]
            results['img_inputs'] = ([imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors],  [imgs2, rots2, trans2, intrins2, post_rots2, post_trans2, bda_rot, gt_depths2, sensor2sensors2] )
        
            if not self.is_train:
                tmp1 = [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors]
                tmp2 = [imgs2, rots2, trans2, intrins2, post_rots2, post_trans2, bda_rot, gt_depths2, sensor2sensors2]

                imgs = torch.cat([tmp1[0], tmp2[0]], dim=0)
                rots = torch.cat([tmp1[1], tmp2[1]], dim=0)
                trans = torch.cat([tmp1[2], tmp2[2]], dim=0)
                intrins = torch.cat([tmp1[3], tmp2[3]], dim=0)
                post_rots = torch.cat([tmp1[4], tmp2[4]], dim=0)
                post_trans = torch.cat([tmp1[5], tmp2[5]], dim=0)
                gt_depths = torch.cat([tmp1[7], tmp2[7]], dim=0)
                sensor2sensors = torch.cat([tmp1[8], tmp2[8]], dim=0)
                # bda_rot = bda_rot[:3, :3]

                results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, imgs.shape[-2:], gt_depths, sensor2sensors)
        
        return results