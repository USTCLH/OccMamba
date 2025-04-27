import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from .lovasz_softmax import lovasz_softmax
from projects.occ_plugin.utils import coarse_to_fine_coordinates, project_points_on_img, coarse_to_fine_coordinates_with_pool_flag
from projects.occ_plugin.utils import nusc_class_frequencies, nusc_class_names
from projects.occ_plugin.utils import semantic_kitti_class_frequencies, kitti_class_names
from projects.occ_plugin.utils import semantic_poss_class_frequencies, poss_class_names
from projects.occ_plugin.utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.occ_plugin.utils import MixerModel, MixerModelForSegmentation
from projects.occ_plugin.utils import H2HE_order_index_within_range
from timm.models.layers import DropPath


@HEADS.register_module()
class OccMamba_Head(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        input_voxel_size=[128, 128, 10],
        trans_dim = 128,
        down_blocks = [2, 2, 2, 2],
        up_blocks = [2, 2, 2, 2],
        order_method = [{'order':'H2HE', 'coor_order':'xy', 'inverse':False}],
        drop_path = 0.2,
        with_lcp = False,
        lcp_sizes = [3, 5, 7],
        lcp_stride = [3, 5, 7],
        lcp_mamba_channel = 128,
        lcp_mamba_expand = 2,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        fine_topk=20000,
        final_occ_size=[512, 512, 40],
        empty_idx=0,
        balance_cls_weight=True,
        cascade_ratio=4,
        upsample_ratio=1,
        out_refine=0,
        refine_method = 'threshold',
        train_cfg=None,
        test_cfg=None,
        dataset='nuscenes',
        **kwargs,
    ):
        super(OccMamba_Head, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fine_topk = fine_topk
        
        self.input_voxel_size = input_voxel_size
        self.final_occ_size = final_occ_size
        self.cascade_ratio = cascade_ratio
        self.upsample_ratio = upsample_ratio
        self.out_refine = out_refine
        self.refine_method = refine_method
        self.dataset = dataset

        # mamba
        self.trans_dim = trans_dim
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.drop_path = drop_path
        self.rms_norm = False
        self.fused_add_norm = False
        self.with_lcp = with_lcp
        self.lcp_sizes = lcp_sizes
        self.lcp_stride = lcp_stride
        self.lcp_mamba_channel = lcp_mamba_channel
        self.lcp_mamba_expand = lcp_mamba_expand

        self.fine_input_dim = 128
        if self.cascade_ratio != 1: 
            self.fine_mlp = nn.Sequential(
                nn.Linear(self.fine_input_dim, 64),
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.out_channels))       
        elif self.upsample_ratio != 1:
            self.fine_out = nn.Sequential(
                build_conv_layer(conv_cfg, self.fine_input_dim, 
                        64, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, 64)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, 64, 
                        self.out_channels, kernel_size=1, stride=1, padding=0))
            if self.out_refine != 0:
                self.fine_out_refine = nn.Sequential(
                    build_conv_layer(conv_cfg, self.in_channels, 
                            64, kernel_size=1, stride=1, padding=0),
                    build_norm_layer(norm_cfg, 64)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, 64, 
                            self.out_channels, kernel_size=1, stride=1, padding=0))


        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        # voxel-level prediction
        self.new_indices = {}

        self.order_method = []
        for method in order_method:
            if method['order'] == 'H2HE':
                fun = partial(H2HE_order_index_within_range, coor_order=method['coor_order'], inverse=method['inverse'])
            else:
                raise NotImplementedError("This reorder method is not implemented yet")
            self.order_method.append(fun)

        self.encoder = nn.Sequential(build_conv_layer(conv_cfg, in_channels=self.in_channels + 3,    # +3 for xyz channels
                                        out_channels=self.trans_dim, kernel_size=1),      
                                     build_norm_layer(norm_cfg, self.trans_dim)[1],
                                     nn.ReLU(inplace=True),
                                     build_conv_layer(conv_cfg, in_channels=self.trans_dim, 
                                        out_channels=self.trans_dim, kernel_size=1))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.blocks = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.upconv = nn.ModuleList()

        self.n_blocks = self.down_blocks + self.up_blocks
        mamba_block_channels = int(self.trans_dim / len(self.order_method))
        for i in range(len(self.n_blocks)):

            self.blocks.append(MixerModel(d_model=mamba_block_channels,
                                n_layer=self.n_blocks[i],
                                rms_norm=self.rms_norm,
                                fused_add_norm=self.fused_add_norm,
                                drop_path=self.drop_path))
            self.norm.append(nn.LayerNorm(mamba_block_channels))

            if i < (len(self.down_blocks)-1):
                self.downsample.append(nn.Sequential(
                                build_conv_layer(conv_cfg, in_channels=self.trans_dim, 
                                        out_channels=self.trans_dim, kernel_size=3, stride=2, padding=1),
                                build_norm_layer(norm_cfg, self.trans_dim)[1],
                                nn.ReLU(inplace=True)))
            if i > (len(self.down_blocks)-1) and i < (len(self.n_blocks)-1):
                self.upconv.append(nn.Sequential(
                            build_conv_layer(conv_cfg, in_channels=self.trans_dim * 2, 
                                    out_channels=self.trans_dim, kernel_size=3, stride=1, padding=1),
                            build_norm_layer(norm_cfg, self.trans_dim)[1],
                            nn.ReLU(inplace=True)))

        self.mamba_out = nn.Sequential()
        mamba_out_channel = self.trans_dim
        while mamba_out_channel > self.fine_input_dim:
            next_channel = max(mamba_out_channel // 2, self.fine_input_dim)
            self.mamba_out.append(nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mamba_out_channel, 
                        out_channels=next_channel, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, next_channel)[1],
                nn.ReLU(inplace=True)))
            mamba_out_channel = next_channel

        if self.with_lcp:
            lcp_layer_num = 0
            self.lcp_mamba = nn.ModuleList()
            self.lcp_linear = nn.ModuleList()
            for i in range(len(self.lcp_sizes)):
                lcp_layer_num += (self.lcp_sizes[i] / self.lcp_stride[i])
                if self.fine_input_dim != self.lcp_mamba_channel:
                    self.lcp_linear.append(nn.Linear(self.fine_input_dim, self.lcp_mamba_channel))
                self.lcp_mamba.append(MixerModel(d_model=self.lcp_mamba_channel,
                                                    n_layer=2,
                                                    ssm_cfg={"expand": self.lcp_mamba_expand},
                                                    rms_norm=self.rms_norm,
                                                    fused_add_norm=self.fused_add_norm,
                                                    drop_path=self.drop_path))
            self.lcp_out = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.lcp_mamba_channel * int(lcp_layer_num), 
                        out_channels=self.fine_input_dim, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, self.fine_input_dim)[1],
                nn.ReLU(inplace=True))

        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.fine_input_dim, 
                        out_channels=64, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, 64)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=64, 
                        out_channels=out_channels, kernel_size=1, stride=1, padding=0))            

        # loss functions
        if balance_cls_weight and self.dataset == 'nuscenes':
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        elif balance_cls_weight and self.dataset == 'kitti':
            self.class_weights = torch.from_numpy(1 / np.log(semantic_kitti_class_frequencies + 0.001))
        elif balance_cls_weight and self.dataset == 'poss':
            self.class_weights = torch.from_numpy(1 / np.log(semantic_poss_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

        if self.dataset == 'nuscenes':
            self.class_names = nusc_class_names 
        elif self.dataset == 'kitti':
            self.class_names = kitti_class_names
        else:
            self.class_names = poss_class_names 

        self.empty_idx = empty_idx

    def reorder_features(self, voxel_feats, order_method=None, new_indices=None):
        B, C, W, H, D = voxel_feats.shape

        features = voxel_feats.permute(0, 2, 3, 4, 1).reshape(B, W*H*D, C)

        x = torch.arange(W, device=features.device).view(W, 1, 1).repeat(1, H, D)
        y = torch.arange(H, device=features.device).view(1, H, 1).repeat(W, 1, D)
        z = torch.arange(D, device=features.device).view(1, 1, D).repeat(W, H, 1)
        coors = torch.stack((x, y, z), dim=0)
        coors = coors.permute(1, 2, 3, 0).reshape(W*H*D, 3).repeat(B, 1, 1).float().cuda()
            
        if new_indices == None:
            new_indices = []
            for method in order_method:
                indices_reordered = method(W, H, D).unsqueeze(0).cuda()
                new_indices.append(indices_reordered)

        # 使用重排索引来重排features和coors
        features_list = []
        coors_list = []
        C_step = int(C / len(new_indices))
        for i, indices in enumerate(new_indices):
            C_start = int(C_step * i)
            C_end = int(C_step * (i+1)) if (i+1) != len(new_indices) else C
            features_list.append(torch.gather(features[:, :, C_start:C_end], 1, indices.unsqueeze(-1).expand(-1, -1, C_step)))
            coors_list.append(torch.gather(coors, 1, indices.unsqueeze(-1).expand(-1, -1, 3)))

        features_reordered = torch.cat(features_list, dim=1).contiguous()
        coors_reordered = torch.cat(coors_list, dim=1).contiguous()

        return features_reordered, coors_reordered, new_indices
    
    def resume_features(self, features_reordered, indices_list, shape):
        B, C, W, H, D = shape
        _, L, C_out = features_reordered.shape

        voxel_feats_list = []
        L_step = int(L / len(indices_list))
        for i, indices in enumerate(indices_list): 
            L_start = int(L_step * i)
            L_end = int(L_step * (i+1)) if (i+1) != len(indices_list) else L
            _, inverse_indices = indices.sort(dim=1)
            features = torch.gather(features_reordered[:, L_start:L_end, :], 1, inverse_indices.unsqueeze(-1).expand(-1, -1, C_out)).contiguous()
            voxel_feats_list.append(features.reshape(B, W, H, D, C_out).permute(0, 4, 1, 2, 3).contiguous())

        voxel_feats = torch.cat(voxel_feats_list, dim=1).contiguous()
        
        return voxel_feats

    def patch_features(self, voxel_feats, lcp_size, shift=0):
        B, C, W, H, D = voxel_feats.shape

        W_shift = W + shift
        H_shift = H + shift
        W_pad = lcp_size * math.ceil(W_shift / lcp_size)
        H_pad = lcp_size * math.ceil(H_shift / lcp_size)

        pad_feats = torch.zeros((B, C, W_pad, H_pad, D), device=voxel_feats.device)
        pad_feats[:, :, shift:W_shift, shift:H_shift, :] = voxel_feats

        patch_feats = pad_feats.unfold(2, lcp_size, lcp_size).unfold(3, lcp_size, lcp_size).contiguous()

        N = (W_pad // lcp_size) * (H_pad // lcp_size)
        L = lcp_size * lcp_size * D
        patch_feats = patch_feats.view(B, C, N, L)

        return patch_feats.contiguous()

    def resume_patch(self, patch_feats, voxel_shape, lcp_size, shift=0):
        _, _, N, L = patch_feats.shape
        B, C, W, H, D = voxel_shape

        W_shift = W + shift
        H_shift = H + shift
        W_pad = lcp_size * math.ceil(W_shift / lcp_size)
        H_pad = lcp_size * math.ceil(H_shift / lcp_size)

        patch_feats = patch_feats.view(B, C, W_pad // lcp_size, H_pad // lcp_size, D, lcp_size, lcp_size)
        patch_feats = patch_feats.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
        patch_feats = patch_feats.view(B, C, W_pad, H_pad, D)

        voxel_feats = patch_feats[:, :, shift:W_shift, shift:H_shift, :]

        return voxel_feats.contiguous()

    def forward_coarse_voxel(self, voxel_feats):
        output = {}

        # encoder and position embedding
        B, C, W, H, D = voxel_feats.shape
        x = torch.arange(W, device=voxel_feats.device).view(W, 1, 1).repeat(1, H, D)
        y = torch.arange(H, device=voxel_feats.device).view(1, H, 1).repeat(W, 1, D)
        z = torch.arange(D, device=voxel_feats.device).view(1, 1, D).repeat(W, H, 1)
        coors = torch.stack((x, y, z), dim=0)
        coors = coors.permute(1, 2, 3, 0).repeat(B, 1, 1, 1, 1).float().cuda().contiguous()

        voxel_feats = torch.cat([voxel_feats, coors.permute(0, 4, 1, 2, 3)], dim=1).contiguous()
        voxel_feats = self.encoder(voxel_feats)

        pos = self.pos_embed(coors).permute(0, 4, 1, 2, 3)
        voxel_feats = voxel_feats + pos
        voxel_feats = voxel_feats.contiguous()

        # mamba blocks
        shapes = []
        features_list = []
        for i in range(len(self.n_blocks)):
            shapes.append(voxel_feats.shape)

            # reorder features (B, W*H*D, C)
            if i in self.new_indices.keys():
                in_features, _, indices_list = self.reorder_features(voxel_feats, new_indices=self.new_indices[i])  
            else:
                in_features, _, indices_list = self.reorder_features(voxel_feats, order_method=self.order_method)
                self.new_indices[i] = indices_list

            # mamba block
            mamba_features = self.blocks[i](in_features)        # B, W*H*D, C_out
            mamba_features = self.norm[i](mamba_features)

            # resume features       
            voxel_feats = self.resume_features(mamba_features, indices_list, shapes[i])
            features_list.append(voxel_feats)

            # downsample, upsample, skip connect
            if i < (len(self.down_blocks)-1):
                voxel_feats = self.downsample[i](voxel_feats)
            elif i > (len(self.down_blocks)-1) and i < (len(self.n_blocks)-1):
                B, C, W, H, D = shapes[(2*(len(self.down_blocks)-1)-i)]
                voxel_feats = F.interpolate(voxel_feats, size=[W, H, D], mode='trilinear', align_corners=False).contiguous()
                voxel_feats = torch.cat([voxel_feats, features_list[(2*(len(self.down_blocks)-1)-i)]], dim=1)
                voxel_feats = self.upconv[i - len(self.down_blocks)](voxel_feats)

        out_voxel_feats = features_list[-1]
        out_voxel_feats = self.mamba_out(out_voxel_feats)

        if self.with_lcp:
            B, C, W, H, D = out_voxel_feats.shape
            out_lcp_feats = []
            for i, lcp_size in enumerate(self.lcp_sizes):
                stride = self.lcp_stride[i]
                stride_num = int(lcp_size / stride)

                # (B C W H D) to (B C N L)
                N_list = []
                lcp_feats_list = []
                for j in range(stride_num):
                    lcp_local_feats = self.patch_features(out_voxel_feats, lcp_size, shift=int(stride * j))
                    N_list.append(lcp_local_feats.shape[2])
                    lcp_feats_list.append(lcp_local_feats)
                lcp_feats = torch.cat(lcp_feats_list, dim=2).contiguous()

                # B C N L to B*N L C'
                _, _, N, L = lcp_feats.shape
                lcp_feats = lcp_feats.permute(0, 2, 3, 1).contiguous().view(B*N, L, C).contiguous()
                if i < len(self.lcp_linear):
                    lcp_feats = self.lcp_linear[i](lcp_feats)
                lcp_feats = self.lcp_mamba[i](lcp_feats)

                # (B*N L C') to (B C' N L) to (B C' W H D)
                _, _, C_new = lcp_feats.shape
                lcp_feats = lcp_feats.view(B, N, L, C_new).permute(0, 3, 1, 2).contiguous()
                N_start = 0
                for j in range(stride_num):
                    lcp_local_feats = lcp_feats[:, :, N_start:(N_start+N_list[j]), :]
                    lcp_local_feats = self.resume_patch(lcp_local_feats, (B, C_new, W, H, D), lcp_size, shift=int(stride * j))
                    out_lcp_feats.append(lcp_local_feats)
                    N_start = N_start + N_list[j]
                # out_lcp_feats.append(lcp_feats)

            out_lcp_feats = torch.cat(out_lcp_feats, dim=1).contiguous()
            out_voxel_feats = self.lcp_out(out_lcp_feats)

        out_voxel = self.occ_pred_conv(out_voxel_feats)

        return {'occ': [out_voxel],
                'out_voxel_feats': [out_voxel_feats],
        }
     
    def forward(self, voxel_feats, **kwargs):
        if type(voxel_feats) is dict:
            voxel_feats = voxel_feats['data']
        
        # assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        # forward voxel 
        output = self.forward_coarse_voxel(voxel_feats)

        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        if self.cascade_ratio != 1:
            coarse_occ_mask = coarse_occ.argmax(1) != self.empty_idx
            assert coarse_occ_mask.sum() > 0, 'no foreground in coarse voxel'

            B, W, H, D = coarse_occ_mask.shape
            coarse_coord_x, coarse_coord_y, coarse_coord_z = torch.meshgrid(torch.arange(W).to(coarse_occ.device),
                        torch.arange(H).to(coarse_occ.device), torch.arange(D).to(coarse_occ.device), indexing='ij')
            
            output['fine_output'] = []
            output['fine_coord'] = []

            for b in range(B):
                this_coarse_coord = torch.stack([coarse_coord_x[coarse_occ_mask[b]],
                                                coarse_coord_y[coarse_occ_mask[b]],
                                                coarse_coord_z[coarse_occ_mask[b]]], dim=0)  # 3, N
                if self.training:
                    this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio, topk=self.fine_topk)  # 3, 8N/64N
                else:
                    this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio)  # 3, 8N/64N
                return_fine_coord = this_fine_coord

                output['fine_coord'].append(return_fine_coord)
                new_coord = this_fine_coord[None].permute(0,2,1).float().contiguous()  # x y z

                this_fine_coord = this_fine_coord.float()
                this_fine_coord[0, :] = (this_fine_coord[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                this_fine_coord[1, :] = (this_fine_coord[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                this_fine_coord[2, :] = (this_fine_coord[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                this_fine_coord = this_fine_coord[None,None,None].permute(0,4,1,2,3).float()
                # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                new_feat = F.grid_sample(out_voxel_feats[b:b+1].permute(0,1,4,3,2), this_fine_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
                fine_feats = new_feat[0,:,:,0,0].permute(1,0)
                assert torch.isnan(new_feat).sum().item() == 0
                
                fine_out = self.fine_mlp(fine_feats)
                output['fine_output'].append(fine_out)

        elif self.upsample_ratio != 1:
            B, C, W, H, D = out_voxel_feats.shape
            new_W, new_H, new_D = W*self.upsample_ratio, H*self.upsample_ratio, D*self.upsample_ratio

            fine_voxel_feats = F.interpolate(out_voxel_feats, size=[new_W, new_H, new_D], mode='trilinear', align_corners=False).contiguous()
            fine_out = self.fine_out(fine_voxel_feats)

            if self.out_refine != 0:
                refine_voxel_feats = F.interpolate(voxel_feats, size=[new_W, new_H, new_D], mode='trilinear', align_corners=False).contiguous()
                refine_out = self.fine_out_refine(refine_voxel_feats)

                refine_prob = F.softmax(fine_out, dim=1)
                refine_max_prob, _ = torch.max(refine_prob, dim=1) 

                if self.refine_method == 'threshold':
                    refine_threshold = self.out_refine
                elif self.refine_method == 'ratio':
                    refine_values, _ = torch.topk(refine_max_prob.view(-1), int(new_W*new_H*new_D*self.out_refine))
                    refine_threshold = refine_values[-1]
                else:
                    refine_threshold = self.out_refine
                
                refine_mask = (refine_max_prob > refine_threshold)
                refine_mask = refine_mask.unsqueeze(1).expand_as(refine_prob)

                refine_out = refine_out + fine_out
                fine_out = torch.where(refine_mask, fine_out, refine_out)

            output['fine_output']= [fine_out]

        return {'output_voxels': output['occ'],
                'output_voxels_fine': output.get('fine_output', None),
                'output_coords_fine': output.get('fine_coord', None),
        }


    def loss_voxel(self, output_voxels, target_voxels, tag):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict

    def loss_point(self, fine_coord, fine_output, target_voxels, tag):

        selected_gt = target_voxels[:, fine_coord[0,:], fine_coord[1,:], fine_coord[2,:]].long()[0]
        assert torch.isnan(selected_gt).sum().item() == 0, torch.isnan(selected_gt).sum().item()
        assert torch.isnan(fine_output).sum().item() == 0, torch.isnan(fine_output).sum().item()

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(fine_output, selected_gt, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(fine_output, dim=1), selected_gt, ignore=255)

        return loss_dict

    def loss(self, output_voxels=None, output_coords_fine=None, output_voxels_fine=None, target_voxels=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='c_{}'.format(index)))
        if self.cascade_ratio != 1:
            loss_batch_dict = {}
            for index, (fine_coord, fine_output) in enumerate(zip(output_coords_fine, output_voxels_fine)):
                this_batch_loss = self.loss_point(fine_coord, fine_output, target_voxels, tag='fine')
                for k, v in this_batch_loss.items():
                    if k not in loss_batch_dict:
                        loss_batch_dict[k] = v
                    else:
                        loss_batch_dict[k] = loss_batch_dict[k] + v
            for k, v in loss_batch_dict.items():
                loss_dict[k] = v / len(output_coords_fine)
        elif self.upsample_ratio != 1:
            for index, output_voxel_fine in enumerate(output_voxels_fine):
                loss_dict.update(self.loss_voxel(output_voxel_fine, target_voxels,  tag='f_{}'.format(index)))

        return loss_dict
