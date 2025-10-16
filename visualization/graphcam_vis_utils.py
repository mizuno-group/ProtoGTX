#!/usr/bin/env python3
"""
Created on 2025-10-16 (Thu) 18:07:16

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import os
import cv2
import h5py
import openslide
import numpy as np

from glob import glob

import torch
import torch.nn as nn


class VisGraphCAM():
    def __init__(self, wsi_dirs, h5_feats_dirs, graphcam_path, vis_save_dir, class_dict={0:'normal', 1:'luad', 2:'lscc'},stack_auto=False):
        self.wsi_dirs = wsi_dirs
        self.h5_feats_dirs = h5_feats_dirs
        self.graphcam_path = graphcam_path
        self.vis_save_dir = vis_save_dir
        self.class_dict = class_dict
        self.stack_auto = stack_auto

    def visualize_graphcam(self, sample_batched, n_class=3, ind=0):
        slide_id = sample_batched['id'][ind].split('.')[0]
        label_id = int(sample_batched['label'][0])
        label_name = [k for k,v in self.class_dict.items() if v==label_id][0]

        h5_feats_fpath = None
        for d in self.h5_feats_dirs:
            tmp_paths = glob(f'{d}/*.h5')
            for path in tmp_paths:
                if slide_id in path:
                    h5_feats_fpath = path
                    break
            if h5_feats_fpath is not None:
                break
        wsi_path = None
        for d in self.wsi_dirs:
            tmp_paths = glob(f'{d}/*')
            for path in tmp_paths:
                if slide_id in path:
                    wsi_path = path
                    break
            if wsi_path is not None:
                break


        slide_id = sample_batched['id'][ind]

        wsi = openslide.open_slide(wsi_path)
        coords = h5py.File(h5_feats_fpath, 'r')['coords'][:]

        p = torch.load(self.graphcam_path+'/prob.pt').cpu().detach().numpy()[0]

        width, height = wsi.dimensions

        w_r, h_r = int(width/20), int(height/20)
        resized_img = wsi.get_thumbnail((w_r,h_r))
        resized_img = resized_img.resize((w_r,h_r))


        output_img = np.asarray(resized_img)[:,:,::-1].copy()
        print('visualize GraphCAM')
        assign_matrix = torch.load(self.graphcam_path+'/s_matrix_ori.pt')
        m = nn.Softmax(dim=1)
        assign_matrix = m(assign_matrix)

        # Thresholding for better visualization
        p = np.clip(p, 0.4, 1)  # NOTE: default (0.4, 1)

        vis_merge_list = []
        for cam_id in range(n_class):
            vis = self.cam_processing(output_img, assign_matrix, p, coords, width, height, cam_id=cam_id)
            os.makedirs(self.vis_save_dir+f'{slide_id}', exist_ok=True)
            cv2.imwrite(self.vis_save_dir+f'{slide_id}/{slide_id}_{label_name}_cam_{cam_id}.png', vis)
            vis_merge_list.append(vis)
        vis_merge = cv2.hconcat([output_img]+vis_merge_list)

        h, w, _ = output_img.shape
        if self.stack_auto:
            if h > w:
                vis_merge = cv2.hconcat([output_img]+vis_merge_list)
            else:
                vis_merge = cv2.vconcat([output_img]+vis_merge_list)
        else:
            vis_merge = cv2.hconcat([output_img]+vis_merge_list)

        cv2.imwrite(self.vis_save_dir+f'{slide_id}/{slide_id}_{label_name}_all_types_cam_all.png', vis_merge)
        cv2.imwrite(self.vis_save_dir+f'{slide_id}/{slide_id}_{label_name}_all_types_ori.png', output_img)


    def cam_processing(self, output_img, assign_matrix, p, coords, width, height, cam_id=0):
        output_img_copy =np.copy(output_img)
        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        image_transformer_attribution = (output_img_copy - output_img_copy.min()) / (output_img_copy.max() - output_img_copy.min())

        cam_matrix = torch.load(self.graphcam_path+f'/cam_{cam_id}.pt')
        cam_matrix = torch.mm(assign_matrix, cam_matrix.transpose(1,0))
        cam_matrix = cam_matrix.cpu()

        # Normalize the graphcam
        cam_matrix = (cam_matrix - cam_matrix.min()) / (cam_matrix.max() - cam_matrix.min())
        cam_matrix = cam_matrix.detach().numpy()
        cam_matrix = p[cam_id] * cam_matrix
        cam_matrix = np.clip(cam_matrix, 0, 1)

        mask = cam_to_mask_absolute(gray=gray, 
                                    coords=coords, 
                                    cam_matrix=cam_matrix, 
                                    wsi_width=width, 
                                    wsi_height=height, 
                                    patch_size=512, 
                                    grid_size=20)
        vis = show_cam_on_image(image_transformer_attribution, mask)
        vis =  np.uint8(255 * vis) 

        return vis

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def cam_to_mask_absolute(gray, coords, cam_matrix, wsi_width, wsi_height, patch_size=512, grid_size=20):
    """
    絶対座標とCAM行列から、grayサイズのマスク画像を生成する
    gray: (h_r, w_r) サイズの縮小画像
    coords: [[x1, y1], [x2, y2], ...] 画素単位の絶対座標
    cam_matrix: CAM値 (N, 1) or (N,)
    wsi_width, wsi_height: WSIのサイズ (例: 33863, 35842)
    patch_size: 各パッチのサイズ (例: 512)
    grid_size: gray上での分割数 (例: 20)
    """
    h_r, w_r = gray.shape
    mask = np.zeros((h_r, w_r), dtype=np.float32)

    # 縮小倍率: gray ← WSI のスケール
    scale_x = w_r / wsi_width
    scale_y = h_r / wsi_height

    for i, (x, y) in enumerate(coords):
        # 左上座標をgrayスケールへ変換
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)

        # パッチサイズをgrayスケールで換算
        x2 = int((x + patch_size) * scale_x)
        y2 = int((y + patch_size) * scale_y)

        if y2 >= h_r or x2 >= w_r or x1 < 0 or y1 < 0:
            continue

        value = cam_matrix[i][0] if cam_matrix.ndim == 2 else cam_matrix[i]
        mask[y1:y2, x1:x2].fill(value)

    return mask