import numpy as np
import torch
import yaml, os

from mmdet.datasets.builder import PIPELINES
from torch.utils import data


@PIPELINES.register_module()
class CreateDepthFromLiDAR(object):
    def __init__(self):
        return
        
    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # the intrinsic matrix is [4, 4] for kitti and [3, 3] for nuscenes 
        if intrins.shape[-1] == 4:
            points = torch.cat((points, torch.ones((points.shape[0], 1, 1, 1))), dim=2)
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

    def __call__(self, results):
        ################################################ img 1
        # loading LiDAR points
        lidar_points = results['origin_points'].tensor[:, :3]
        
        # project voxels onto the image plane
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][0][:6]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)[:, 0]
        
        # create depth map
        img_h, img_w = imgs[0].shape[-2:]
        valid_mask = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 0] <= img_w - 1) & \
                    (projected_points[:, 1] <= img_h - 1) & \
                    (projected_points[:, 2] > 0)

        # create projected depth map
        img_depth = torch.zeros((img_h, img_w))
        depth_projected_points = projected_points[valid_mask]
        # sort and project
        depth_order = torch.argsort(depth_projected_points[:, 2], descending=True)
        depth_projected_points = depth_projected_points[depth_order]
        img_depth[depth_projected_points[:, 1].round().long(), depth_projected_points[:, 0].round().long()] = depth_projected_points[:, 2]
        
        imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors = results['img_inputs'][0]
        tmp1 = [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, imgs.shape[-2:], img_depth.unsqueeze(0), sensor2sensors]
        
        ################################################ img 2
        # loading LiDAR points
        lidar_points = results['origin_points'].tensor[:, :3]

        # project voxels onto the image plane
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][1][:6]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)[:, 0]
        
        # create depth map
        img_h, img_w = imgs[0].shape[-2:]
        valid_mask = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 0] <= img_w - 1) & \
                    (projected_points[:, 1] <= img_h - 1) & \
                    (projected_points[:, 2] > 0)
        
        # create projected depth map
        img_depth = torch.zeros((img_h, img_w))
        depth_projected_points = projected_points[valid_mask]
        # sort and project
        depth_order = torch.argsort(depth_projected_points[:, 2], descending=True)
        depth_projected_points = depth_projected_points[depth_order]
        img_depth[depth_projected_points[:, 1].round().long(), depth_projected_points[:, 0].round().long()] = depth_projected_points[:, 2]  
  
        imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors = results['img_inputs'][1]
        tmp2 = [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, imgs.shape[-2:], img_depth.unsqueeze(0), sensor2sensors]

        ################################################ 
        imgs = torch.cat([tmp1[0], tmp2[0]], dim=0)
        rots = torch.cat([tmp1[1], tmp2[1]], dim=0)
        trans = torch.cat([tmp1[2], tmp2[2]], dim=0)
        intrins = torch.cat([tmp1[3], tmp2[3]], dim=0)
        post_rots = torch.cat([tmp1[4], tmp2[4]], dim=0)
        post_trans = torch.cat([tmp1[5], tmp2[5]], dim=0)
        gt_depths = torch.cat([tmp1[8], tmp2[8]], dim=0)
        sensor2sensors = torch.cat([tmp1[9], tmp2[9]], dim=0)
        # bda_rot = bda_rot[:3, :3]

        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, imgs.shape[-2:], gt_depths, sensor2sensors)
    
        return results
