_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"
img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]  # [128 128 10]
cascade_ratio = 4
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

# mamba setting
mamba_trans_dim = 384
mamba_down_blocks = [2, 2, 2, 2]
mamba_up_blocks = [2, 2, 2, 2]
mamba_order_method = [{'order':'H2HE', 'coor_order':'xy', 'inverse':False}]    #  Height-prioritized 2D Hilbert expansion

with_lcp = True
lcp_sizes = [3, 5, 7]
lcp_stride = [3, 5, 7]
lcp_mamba_channel = 256
lcp_mamba_expand = 1

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='OccMamba',
    loss_norm=True,
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoderHD',
        in_channels=5,
        sparse_shape=[81, 1024, 1024],
        norm_cfg=dict(type='SyncBN', requires_grad=True), 
        output_channels=numC_Trans,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        encoder_strides=(2, 2, 2, 1), 
        block_type='basicblock',
        fp16_enabled=False,
    ), # not enable FP16 here
    pts_bbox_head=dict(
        type='OccMamba_Head',
        in_channels=numC_Trans,  
        out_channels=num_cls,
        trans_dim = mamba_trans_dim,
        down_blocks = mamba_down_blocks,
        up_blocks = mamba_up_blocks,
        order_method = mamba_order_method,
        with_lcp = with_lcp,
        lcp_sizes = lcp_sizes,
        lcp_stride = lcp_stride,
        lcp_mamba_channel = lcp_mamba_channel,
        lcp_mamba_expand = lcp_mamba_expand,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        cascade_ratio=cascade_ratio,
        final_occ_size=occ_size,
        fine_topk=10000,
        empty_idx=empty_idx,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['gt_occ', 'points'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]


test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR'),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(
    interval=20,
    pipeline=test_pipeline,
    # save_best='SSC_mean',
    save_best='SSC_fine_mean',
    rule='greater',
)

# custom_hooks = [
#     dict(type='OccEfficiencyHook'),
# ]

# find_unused_parameters = True
