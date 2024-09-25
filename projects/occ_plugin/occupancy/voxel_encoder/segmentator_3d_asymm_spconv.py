# -*- coding:utf-8 -*-
# author: Xinge, Xzy
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn
from mmdet3d.models.builder import MIDDLE_ENCODERS

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + '1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + '2')
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + '3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        upA = self.up_subm(upA)
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))

        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut



def extract_nonzero_features(x):
    device = x.device
    nonzero_index = torch.sum(torch.abs(x), dim=1).nonzero()
    coords = nonzero_index.type(torch.int32).to(device)
    channels = int(x.shape[1])
    features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    features = features[torch.sum(torch.abs(features), dim=1).nonzero(), :]
    features = features.squeeze(1).to(device)
    coords, _, _ = torch.unique(coords, return_inverse=True, return_counts=True, dim=0)
    return coords, features

class Asymm_3d_spconv_compeletion(nn.Module):
    def __init__(
            self,
            chs=[5, 32, 80],
            mybias=False,
            **kwargs,
        ):
        super(Asymm_3d_spconv_compeletion, self).__init__()

        self.a_conv1 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv2 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv3 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 5, 1, padding=2, bias=mybias), nn.ReLU())
        self.a_conv4 = nn.Sequential(nn.Conv3d(chs[1], chs[1], 7, 1, padding=3, bias=mybias), nn.ReLU())
        self.a_conv5 = nn.Sequential(nn.Conv3d(chs[1]*3, chs[1], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.a_conv6 = nn.Sequential(nn.Conv3d(chs[1]*3, chs[1], 5, 1, padding=2, bias=mybias), nn.ReLU())
        self.a_conv7 = nn.Sequential(nn.Conv3d(chs[1]*3, chs[1], 7, 1, padding=3, bias=mybias), nn.ReLU())
        self.ch_conv1 = nn.Sequential(nn.Conv3d(chs[1]*7, chs[2], kernel_size=1, stride=1, bias=mybias), nn.ReLU())
        self.res_1 = nn.Sequential(nn.Conv3d(chs[1], chs[2], 3, 1, padding=1, bias=mybias), nn.ReLU())
        self.res_2 = nn.Sequential(nn.Conv3d(chs[1], chs[2], 5, 1, padding=2, bias=mybias), nn.ReLU())
        self.res_3 = nn.Sequential(nn.Conv3d(chs[1], chs[2], 7, 1, padding=3, bias=mybias), nn.ReLU())
    
    def forward(self, x_dense):
        x1 = self.a_conv1(x_dense)
        x2 = self.a_conv2(x1)
        x3 = self.a_conv3(x1)
        x4 = self.a_conv4(x1)
        t1 = torch.cat((x2, x3, x4), 1)
        x5 = self.a_conv5(t1)
        x6 = self.a_conv6(t1)
        x7 = self.a_conv7(t1)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), 1)
        y0 = self.ch_conv1(x)
        y1 = self.res_1(x_dense)
        y2 = self.res_2(x_dense)
        y3 = self.res_3(x_dense)
        # x = x_dense + y0 + y1 + y2 + y3
        x = y0 + y1 + y2 + y3

        return x

class Asymm_3d_spconv_segmentation(nn.Module):
    def __init__(
            self,
            chs=[5, 32, 256, 20],
            **kwargs,
        ):
        super(Asymm_3d_spconv_segmentation, self).__init__()

        self.downCntx = ResContextBlock(chs[2], chs[1], indice_key="pre")
        self.resBlock2 = ResBlock(chs[1], chs[1]*2, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(chs[1]*2, chs[1]*4, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(chs[1]*4, chs[1]*8, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(chs[1]*8, chs[1]*16, 0.2, pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = UpBlock(chs[1]*16, chs[1]*16, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(chs[1]*16, chs[1]*8, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(chs[1]*8, chs[1]*4, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(chs[1]*4, chs[1]*2, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(chs[1]*2, chs[1]*2, indice_key="recon")

        self.logits = spconv.SubMConv3d(chs[1]*4, chs[3], indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        x = self.downCntx(x)
        down1c, down1b = self.resBlock2(x)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        x = logits.dense()

        return x

@MIDDLE_ENCODERS.register_module()
class Asymm_3d_spconv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            sparse_shape,
            downsample_layers=3,
            init_size=32,
            compeletion=True,
            segmentation=False,
            **kwargs,
        ):
        super(Asymm_3d_spconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse_shape = np.asarray(sparse_shape)
        self.sparse_shape_2 = (self.sparse_shape / (2**downsample_layers)).astype(np.int32)
        self.downsample_layers = downsample_layers
        self.compeletion = compeletion
        self.segmentation = segmentation

        mybias = False  # False
        if self.compeletion and not self.segmentation:
            chs = [self.in_channels, init_size, self.out_channels]
        elif self.compeletion and self.segmentation:
            chs = [self.in_channels, init_size, init_size*8, self.out_channels]
        else:
            print('not implement error.')

        self.spconv_layers = nn.ModuleList()
        downsample_in_channels = chs[0]
        downsample_out_channels = chs[1]
        for i in range(self.downsample_layers):
            self.spconv_layers.append(spconv.SparseSequential(spconv.SparseConv3d(downsample_in_channels, downsample_out_channels, 
                                    3, stride=2, padding=1, bias=mybias, indice_key=f'spconv{i+1}'), nn.ReLU()))
            downsample_in_channels = downsample_out_channels

        ### Completion sub-network
        if self.compeletion:
            self.completion_layer = Asymm_3d_spconv_compeletion(chs, mybias)

        ### Segmentation sub-network
        if self.segmentation:
            self.segmentation_layer = Asymm_3d_spconv_segmentation(chs)

    def forward(self, voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        x_sparse = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)

        # downsample
        for spconv_conv in self.spconv_layers:
            x_sparse = spconv_conv(x_sparse)

        # Spase to dense
        x_dense = x_sparse.dense() 

        ### Completion sub-network by dense convolution
        if self.compeletion:
            x = self.completion_layer(x_dense)

        ### Segmentation sub-network by sparse convolution
        if self.segmentation:
            # Dense to sparse
            coord, features = extract_nonzero_features(x)
            x = spconv.SparseConvTensor(features, coord.int(), self.sparse_shape_2, batch_size)

            x = self.segmentation_layer(x)

        return {'x': x.permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}
        
    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x