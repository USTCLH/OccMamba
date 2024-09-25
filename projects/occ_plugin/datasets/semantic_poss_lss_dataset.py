import numpy as np
import torch
import glob
import os

from mmdet.datasets import DATASETS
from mmdet3d.datasets import SemanticKITTIDataset

from projects.occ_plugin.utils.formating_kitti import cm_to_ious_kitti, format_SC_results_kitti, format_SSC_results_kitti

@DATASETS.register_module()
class CustomSemanticPOSSLssDataset(SemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, split, occ_size, pc_range, camera_used=None,
                 load_continuous=False, *args, **kwargs):
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        if camera_used is not None:
            self.camera_used = [self.camera_map[camera] for camera in camera_used]
        else:
            self.camera_used = None
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        
        self.load_continuous = load_continuous
        self.splits = {
            "train": ["00", "01", "03", "04", "05"],
            "val": ["02"],
            "trainval": ["00", "01", "02", "03", "04", "05"],
        }
        
        self.sequences = self.splits[split]
        self.n_classes = 20
        super().__init__(*args, **kwargs)

        # add mmdet3d.datasets.custom_3d line87 if_else

        self._set_group_flag()
    
    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out

    @staticmethod
    def read_poses(poses_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        poses = []
        with open(poses_path, 'r') as file:
            for line in file:
                data = list(map(float, line.strip().split()))
                if len(data) == 12:
                    pose = torch.FloatTensor(np.array(data).reshape(3, 4))
                    poses.append(pose)
        return poses

    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            poses = self.read_poses(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            )

            voxel_base_path = os.path.join(self.ann_file, 'voxel_labels', sequence)
            img_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence)

            # for semanticposs, as frame sometimes start at 1.
            if not os.path.exists(os.path.join(img_base_path, 'velodyne', '000000.bin')):
                poses.insert(0, torch.zeros((3, 4)))

            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "dataset", "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                pts_path = os.path.join(img_base_path, 'velodyne', img_id + '.bin')
                pts_label_path = os.path.join(img_base_path, 'labels', img_id + '.label')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')

                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None

                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "pts_filename": pts_path,
                        "pts_label_path": pts_label_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        "poses": poses,
                    })
                
        return scans  # return to self.data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        # init for pipeline
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']

        if info is None:
            print('voxel_path', 'of', self.data_infos[index]['pts_filename'], 'found None')
            return None

        return np.load(info)
        

    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
            "poses": poses,
            'pts_filename': pts_filename,
            'pts_label_path': pts_label_path,
            'gt_occ': gt_occ,
        '''
        
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
            poses = info['poses']
        )
        
        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        
        if self.camera_used is not None:
            for cam_type in self.camera_used:
                image_paths.append(info['img_{}_path'.format(int(cam_type))])
                lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
                cam_intrinsics.append(info['P{}'.format(int(cam_type))])
                lidar2cam_rts.append(info['T_velo_2_cam'])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        input_dict['pts_filename'] = info['pts_filename']
        input_dict['pts_label_path'] = info['pts_label_path']

        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index)

        return input_dict
        
    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious_kitti(evaluation_semantic)
        res_table, res_dic = format_SC_results_kitti(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious_kitti(evaluation_semantic)
        res_table, res_dic = format_SSC_results_kitti(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious_kitti(evaluation_semantic)
            res_table, res_dic = format_SSC_results_kitti(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results