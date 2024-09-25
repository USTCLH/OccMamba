# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter

from mmdet3d.models.builder import VOXEL_ENCODERS

@VOXEL_ENCODERS.register_module()
class cylinder_fea(nn.Module):
    def __init__(self, num_features, out_pt_fea_dim=64, fea_compre=None, **kwargs):
        super(cylinder_fea, self).__init__()

        self.num_features = num_features
        self.fea_compre = fea_compre
        self.out_pt_fea_dim = out_pt_fea_dim

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(self.num_features),

            nn.Linear(self.num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, self.out_pt_fea_dim)
        )

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.out_pt_fea_dim, self.fea_compre),
                nn.ReLU())

    # def forward(self, pt_fea, xy_ind):
    def forward(self, features, num_points, coors, *args, **kwargs):
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1).contiguous()

        ### process feature
        voxel_features = self.PPmodel(points_mean)

        if self.fea_compre:
            voxel_features = self.fea_compression(voxel_features)

        return voxel_features
