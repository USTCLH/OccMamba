import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class ConcatFuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img_voxel_feats, pts_voxel_feats, **kwargs):
        return torch.cat([img_voxel_feats, pts_voxel_feats], dim=1)
