import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def pcd_to_voxel(pcd, voxel_size):
    # 将点云坐标向下取整到最近的栅格点
    voxel = np.floor(pcd[:, 0:3] / voxel_size).astype(int) * voxel_size
    voxel = np.c_[voxel, pcd[:, 3:]]

    # 找出唯一的栅格点
    unique_voxel = np.unique(voxel, axis=0)

    return unique_voxel

def process_file(pred_path, save_sequence_path, idx, voxel_size, occ_size):
    file = os.path.join(pred_path, idx, 'pred_f.npy')
    pred = np.load(file)
    pred = pcd_to_voxel(pred, voxel_size)

    voxel = torch.zeros(occ_size)
    for label in range(int(pred[:, 3].min()), int(pred[:, 3].max()) + 1):
        indices = pred[pred[:, 3] == label]
        if indices.shape[0] > 0:
            voxel[indices[:, 2], indices[:, 1], indices[:, 0]] = label

    np.save(os.path.join(save_sequence_path, idx + '_1_1.npy'), voxel.numpy())

if __name__ == '__main__':  
    pred_path = 'output/' + sys.argv[1]
    save_path = 'output/for_submit/voxel_labels'
    sequences = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    
    occ_size = [256, 256, 32]
    voxel_size = 0.2
    max_workers = 16

    for sequence in sequences:
        pred_sequence_path = os.path.join(pred_path, sequence)
        save_sequence_path = os.path.join(save_path, sequence)
        os.makedirs(save_sequence_path, exist_ok=True)
        idx_list = sorted(os.listdir(pred_sequence_path))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, pred_sequence_path, save_sequence_path, idx, voxel_size, occ_size): idx for idx in idx_list}
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()  # 获取结果以捕捉潜在异常
