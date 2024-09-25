
import torch.nn.functional as F
import torch
import numpy as np
from os import path as osp
import os

def save_occ(pred_c, pred_f, points, img_metas, path, visible_mask=None, gt_occ=None, free_id=0, thres_low=0.4, thres_high=0.99):

    """
    visualization saving for paper:
    1. gt
    2. pred_f pred_c
    3. gt visible
    4. pred_f visible
    """
    model_out = pred_f[0].cpu().numpy()
    pred_f = F.softmax(pred_f, dim=1)
    pred_f = pred_f[0].cpu().numpy()  # C W H D
    pred_c = F.softmax(pred_c, dim=1)
    pred_c = pred_c[0].cpu().numpy()  # C W H D
    if visible_mask is not None:
        visible_mask = visible_mask[0].cpu().numpy().reshape(-1) > 0  # WHD
    _, W, H, D = pred_f.shape
    coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
    _, W, H, D = pred_c.shape
    coordinates_3D_c = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
    pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
    pred_c = np.argmax(pred_c, axis=0) # (W, H, D)
    occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
    occ_pred_c_mask = (pred_c.reshape(-1))!=free_id
    pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    if visible_mask is not None:
        pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    else:
        pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    if gt_occ is not None:
        gt_occ = gt_occ.data[0][0].cpu().numpy()  # W H D
        gt_occ[gt_occ==255] = 0
        occ_gt_mask = (gt_occ.reshape(-1))!=free_id
        gt_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
        if visible_mask is not None:
            gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
        else:
            gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls

    if 'scene_token' in img_metas.data[0][0].keys():
        scene_token = img_metas.data[0][0]['scene_token']
        lidar_token = img_metas.data[0][0]['lidar_token']
        save_path = osp.join(path, scene_token, lidar_token)
    else:
        sequence = img_metas.data[0][0]['sequence']
        frame_id = img_metas.data[0][0]['frame_id']
        save_path = osp.join(path, sequence, frame_id)

    if not osp.exists(save_path):
        os.makedirs(save_path)
    save_points_path = osp.join(save_path, 'points.npy')
    save_model_out_path = osp.join(save_path, 'model_out.npy')
    save_pred_f_path = osp.join(save_path, 'pred_f.npy')
    save_pred_c_path = osp.join(save_path, 'pred_c.npy')
    save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
    save_gt_path = osp.join(save_path, 'gt.npy')
    save_gt_v_path = osp.join(save_path, 'gt_visible.npy')

    np.save(save_points_path, points.data[0][0].cpu().numpy())
    np.save(save_pred_f_path, pred_f_save)
    np.save(save_model_out_path, model_out)
    # np.save(save_pred_c_path, pred_c_save)
    # if visible_mask is not None:
    #     np.save(save_pred_f_v_path, pred_f_visible_save)

    if gt_occ is not None:
        np.save(save_gt_path, gt_save)
        # if visible_mask is not None:
            # np.save(save_gt_v_path, gt_visible_save)

