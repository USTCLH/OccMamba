# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.builder import MIDDLE_ENCODERS

try:
    from mmdet3d.ops.spconv import SparseConvTensor, SparseSequential
    print('use mmdet3d spconv')
except ImportError:
    try:
        from spconv.pytorch import SparseConvTensor, SparseSequential
        print('use origin spconv')
    except ImportError:
        from mmcv.ops import SparseConvTensor, SparseSequential
        print('use mmcv spconv')

@MIDDLE_ENCODERS.register_module()
class SparseEncoderHD(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 encoder_strides=(2, 2, 2, 1),
                 block_type='conv_module',
                 keep_depth=True,
                 fp16_enabled=False,
                 **kwargs):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.encoder_strides = encoder_strides
        self.stage_num = len(self.encoder_channels)
        self.keep_depth = keep_depth
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg, 
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,  # GN will result zero
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            self.base_channels,
            norm_cfg=dict(type='GN', num_groups=16, requires_grad=True),
            block_type=block_type)

        # new add
        self.decoder_layers = SparseSequential()

        blocks_list = []
        while encoder_out_channels > self.output_channels * 2:
            blocks_list.append(make_sparse_convmodule(
            encoder_out_channels,
            int(encoder_out_channels / 2),
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            norm_cfg=dict(type='GN', num_groups=16, requires_grad=True), 
            padding=0,
            indice_key=f'decoder{encoder_out_channels}',
            conv_type='SubMConv3d'))

            encoder_out_channels = int(encoder_out_channels / 2)

        blocks_list.append(make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            norm_cfg=dict(type='GN', num_groups=16, requires_grad=True), 
            padding=0,
            indice_key='spconv_down2',
            conv_type='SubMConv3d'))

        stage_name = 'decoder_layer'
        stage_layers = SparseSequential(*blocks_list)
        self.decoder_layers.add_module(stage_name, stage_layers)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)

        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        x = encode_features[-1]

        # new add
        decode_features = []
        for decoder_layers in self.decoder_layers:
            x = decoder_layers(x)
            decode_features.append(x)
        x = decode_features[-1]

        # for detection head
        # [200, 176, 5] -> [200, 176, 5]
        # out = self.conv_out(x)
        out = x
        spatial_features = out.dense()

        if not self.keep_depth:
            spatial_features = spatial_features.sum(dim=2)

        # print(spatial_features.shape)

        return {'x': spatial_features.permute(0,1,4,3,2), # not dense:  B, C, W, H, D 
                'pts_feats': [spatial_features]}

    def make_encoder_layers(self,
                            make_block,
                            in_channels,
                            norm_cfg,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=self.encoder_strides[i],
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                            # conv_type='SubMConv3d'))    # change to this
                elif block_type == 'basicblock':        # this
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=self.encoder_strides[i],
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))     # SubMConv3d
                
                    # define SparseBasicBlock as SBB

                    # conv_input:
                    #     SubMConv3d

                    # encoder_layer1(16):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer2(32):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer3(64):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer4(128):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d)

                    # conv_out:
                    #     SubMConv3d(128->80)


                    # ablation

                    # conv_input:
                    #     SubMConv3d

                    # encoder_layer1(16):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer2(32):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer3(64):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=2)
                    # encoder_layer4(128):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=1)
                    # encoder_layer5(256):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d) -> SparseConv3d(stride=1)
                    # encoder_layer6(512):
                    #     SBB(SubMConv3d) -> SBB(SubMConv3d)

                    # conv_out:
                    #     SubMConv3d(512->256) -> SubMConv3d(256->128) -> SubMConv3d(128->80)
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))    
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
