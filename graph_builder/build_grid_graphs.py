# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 (Thu) 10:35:59

Reference
- https://github.com/vkola-lab/tmi2022/blob/main/feature_extractor/build_graphs.py

@author: I.Azuma
"""

import os
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob

import torch


class CLAMGraphBuilder():
    def __init__(self, patch_dir, feature_dir, qc_info_dir, wsi_dir, save_dir, do_qc=True, wsi_ext = '.svs'):
        self.patch_dir = patch_dir
        self.feature_dir = feature_dir
        self.qc_info_dir = qc_info_dir
        self.wsi_dir = wsi_dir
        self.save_dir = save_dir
        self.wsi_ext = wsi_ext

        self.patch_files = sorted(glob(os.path.join(patch_dir, 'patches/*.h5')))
        self.feature_files = sorted(glob(os.path.join(feature_dir, 'feats_h5/*.h5')))
        self.qc_info_files = sorted(glob(os.path.join(qc_info_dir, '*.h5')))
        wsi_pattern = os.path.join(wsi_dir, '**', f'*{wsi_ext}')
        self.wsi_files = glob(wsi_pattern, recursive=True)
        self.do_qc = do_qc

        print(f"Found {len(self.patch_files)} patch files.")
        print(f"Found {len(self.feature_files)} feature files.")
        print(f"Found {len(self.qc_info_files)} QC info files.")
        print(f"Found {len(self.wsi_files)} WSI files.")
        print(f"QC: {self.do_qc}")


    def build_graphs(self, patch_size=512):
        graphs = []

        file_names = [t.split('/')[-1].split('.h5')[0] for t in self.patch_files]
        for file_name in tqdm(file_names):
            patch_path = os.path.join(self.patch_dir, 'patches', f'{file_name}.h5')
            feature_path = os.path.join(self.feature_dir, 'feats_h5', f'{file_name}.h5')
            qc_info_path = os.path.join(self.qc_info_dir, f'{file_name}_qc_info.h5')
            wsi_path_candidates = [p for p in self.wsi_files if os.path.splitext(os.path.basename(p))[0] == file_name]
            if len(wsi_path_candidates) == 0:
                print(f"WSI file not found for {file_name}, skipping...")
                continue
            if len(wsi_path_candidates) > 1:
                print(f"Multiple WSI files found for {file_name}, using the first one.")

            wsi_path = wsi_path_candidates[0]
            wsi = openslide.open_slide(wsi_path)

            try:
                with h5py.File(feature_path, 'r') as h5:
                    coords = h5['coords'][:]
                    features = h5['features'][:]
                
                if self.do_qc:
                    with h5py.File(qc_info_path, 'r') as qc_info:
                        passed_ids = qc_info['passed_ids'][:]
                        bkg_scores = qc_info['bkg_scores'][:]
                else:
                    passed_ids = np.arange(coords.shape[0])
                    assert len(passed_ids) == features.shape[0], "QC not applied, all patches should be passed."

                if len(passed_ids) == 0:
                    print(f"No passed patches for {file_name}, skipping...")
                    continue
            except:
                print(f"Error reading files for {file_name}, skipping...")
                continue

            assert coords.shape[0] == features.shape[0]

            passed_coords = coords[passed_ids]
            passed_features = features[passed_ids]

            adj = build_adjacency(passed_coords, patch_size=512)

            assert adj.shape[0] == passed_features.shape[0]

            # save adjacency matrix and passed_features
            save_dir = os.path.join(self.save_dir, file_name)
            os.makedirs(save_dir, exist_ok=True)

            adj_save_path = os.path.join(save_dir, f'adj_s_{file_name}.pt')
            torch.save(adj, adj_save_path)

            features_save_path = os.path.join(save_dir, f'features_{file_name}.pt')
            torch.save(torch.from_numpy(passed_features).float(), features_save_path)



def build_adjacency(coords: np.ndarray, patch_size: int = 512) -> torch.Tensor:
    N = coords.shape[0]
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords)}

    # 8 directions for adjacency based on patch size
    directions = [
        (-patch_size, -patch_size), (0, -patch_size), (patch_size, -patch_size),
        (-patch_size, 0),                            (patch_size, 0),
        (-patch_size, patch_size), (0, patch_size), (patch_size, patch_size),
    ]

    adj = np.zeros((N, N), dtype=np.uint8)

    for i, (x, y) in enumerate(coords):
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            j = coord_to_index.get(neighbor)
            if j is not None:
                adj[i, j] = 1
                adj[j, i] = 1  # 対称にする

    return torch.from_numpy(adj).float().cuda()


def visualize_patch_grid(coords, adj, wsi, patch_size=512, patch_level=0, max_examples=3, neighbor_n=5):
    adj = adj.cpu().numpy()
    neighbor_counts = adj.sum(axis=1)
    target_indices = np.where(neighbor_counts == neighbor_n)[0]  # center patch

    # make coordinate to index mapping
    coord_dict = {tuple(coord): idx for idx, coord in enumerate(coords)}

    directions = [
        (-1, -1), ( 0, -1), ( 1, -1),
        (-1,  0), ( 0,  0), ( 1,  0),
        (-1,  1), ( 0,  1), ( 1,  1)
    ]

    for count, center_idx in enumerate(target_indices):
        if count >= max_examples:
            break

        center_coord = tuple(coords[center_idx])
        print(f"\n🧩 Center index: {center_idx}, coord: {center_coord}")

        fig, axs = plt.subplots(3, 3, figsize=(9, 9))
        axs = axs.reshape(3, 3)

        for dx, dy in directions:
            rel_x = dx * patch_size
            rel_y = dy * patch_size
            neighbor_coord = (center_coord[0] + rel_x, center_coord[1] + rel_y)
            ax = axs[dy + 1][dx + 1] 

            if neighbor_coord in coord_dict:
                idx = coord_dict[neighbor_coord]
                img = wsi.read_region(neighbor_coord, patch_level, (patch_size, patch_size)).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"{neighbor_coord}")
            else:
                ax.set_facecolor("lightgray")
                ax.set_title("Missing")

            ax.axis('off')

        plt.suptitle(f"Center patch at {center_coord} (index {center_idx})")
        plt.tight_layout()
        plt.show()
