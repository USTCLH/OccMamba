import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import build_norm_layer


@FUSION_LAYERS.register_module()
class VisFuser(nn.Module):
    def __init__(self,in_channels=None, in_channels_img=None, in_channels_pts=None, out_channels=None, norm_cfg=None) -> None:
        super().__init__()

        assert in_channels is not None or (in_channels_img is not None and in_channels_pts is not None)

        if in_channels is not None:
            self.in_channels_img = in_channels
            self.in_channels_pts = in_channels
        else:
            self.in_channels_img = in_channels_img
            self.in_channels_pts = in_channels_pts

        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(self.in_channels_img, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

        self.pts_enc = nn.Sequential()
        now_channel = self.in_channels_pts
        while now_channel > self.out_channels:
            next_channel = max(now_channel // 2, self.out_channels)
            self.pts_enc.append(nn.Sequential(nn.Conv3d(now_channel, next_channel, 7, padding=3, bias=False),
                                build_norm_layer(norm_cfg, next_channel)[1],
                                nn.ReLU(True)))
            now_channel = next_channel
        if len(self.pts_enc) == 0:
            self.pts_enc.append(nn.Sequential(nn.Conv3d(self.in_channels_pts, self.out_channels, 7, padding=3, bias=False),
                                build_norm_layer(norm_cfg, self.out_channels)[1],
                                nn.ReLU(True)))

        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_voxel_feats, pts_voxel_feats, **kwargs):
        img_voxel_feats = self.img_enc(img_voxel_feats)
        for enc in self.pts_enc:
            pts_voxel_feats = enc(pts_voxel_feats)

        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats

        return voxel_feats

