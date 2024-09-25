import os
import torch
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class LoadPointsFromMultiFrames_kitti(LoadPointsFromFile):
    def __init__(self, start=0, end=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end

    def __call__(self, results):
        results = super().__call__(results)

        current_frame_id = int(results['pts_filename'].split('/')[-1].split('.')[0])
        dir_name = os.path.dirname(results['pts_filename'])

        frame_num = len(results['poses'])
        current_R = results['poses'][current_frame_id][:, :3]
        current_t = results['poses'][current_frame_id][:, 3]

        point_clouds = []
        for i in range(self.start, self.end+1):
            if i == 0:
                continue

            frame_id = current_frame_id + i
            if frame_id < 0 or frame_id >= frame_num:
                continue

            frame_file = os.path.join(dir_name, str(frame_id).zfill(6) + '.bin')
            if not os.path.exists(frame_file):
                continue

            # 创建一个局部的results副本并更新pts_filename
            temp_results = results.copy()
            temp_results['pts_filename'] = frame_file
            point_cloud = super().__call__(temp_results)

            # 根据pose到对应坐标下
            R = results['poses'][frame_id][:, :3]
            t = results['poses'][frame_id][:, 3]
            t[2] = current_t[2]

            points = point_cloud['points'].tensor[:, :3]
            # global_points = torch.mm(points - t, R.T)
            # local_points = torch.mm(global_points, current_R) + current_t
            global_points = points - t
            local_points = global_points + current_t

            point_cloud['points'].tensor[:, :3] = local_points
            point_clouds.append(point_cloud['points'].tensor)

        # 将多个帧的点云数据拼接在一起
        for point_cloud in point_clouds:
            results['points'].tensor = torch.cat([results['points'].tensor, point_cloud], dim=0)

        # import numpy as np
        # np.save('tmp.npy', results['points'].tensor.data.cpu().numpy())
        # asd

        return results