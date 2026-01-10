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


    def build_graphs(self, patch_size=512, overwrite=False, sparse=False):

        file_names = [t.split('/')[-1].split('.h5')[0] for t in self.patch_files]
        for file_name in tqdm(file_names):

            # save adjacency matrix and passed_features
            save_dir = os.path.join(self.save_dir, file_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                if overwrite:
                    print(f"Overwriting existing graph for {file_name}...")
                else:
                    print(f"Graph for {file_name} already exists, skipping...")
                continue

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

            if sparse:
                adj = build_sparse_adjacency(passed_coords, patch_size=patch_size)
            else:
                adj = build_adjacency(passed_coords, patch_size=patch_size)

            assert adj.shape[0] == passed_features.shape[0]

            adj_save_path = os.path.join(save_dir, f'adj_s_{file_name}.pt')
            torch.save(adj, adj_save_path)

            features_save_path = os.path.join(save_dir, f'features_{file_name}.pt')
            torch.save(torch.from_numpy(passed_features).float(), features_save_path)
    
    def symmetric_qc(self, adj_candi=[]):
        #adj_candi = sorted(glob(f"{BASE_DIR}/datasource/TCGA/NSCLC/graphs/251219/*/adj_s_*.pt"))
        for p in tqdm(adj_candi):
            adj = torch.load(p)
            if not is_symmetric_sparse(adj):
                print(f"Not symmetric: {p}")


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
                adj[j, i] = 1  # symmetric

    return torch.from_numpy(adj).float().cuda()

def build_sparse_adjacency(coords: np.ndarray, patch_size: int = 512, device: str = "cpu") -> torch.Tensor:
    coords = np.asarray(coords)
    N = coords.shape[0]
    coord_to_index = { (int(x), int(y)): i for i, (x, y) in enumerate(coords) }

    directions = [
        (-patch_size, -patch_size), (0, -patch_size), (patch_size, -patch_size),
        (-patch_size, 0),                              (patch_size, 0),
        (-patch_size, patch_size),  (0, patch_size),  (patch_size, patch_size),
    ]

    rows = []
    cols = []
    for i, (x, y) in enumerate(coords):
        x = int(x); y = int(y)
        for dx, dy in directions:
            j = coord_to_index.get((x + dx, y + dy))
            if j is not None:
                rows.append(i); cols.append(j)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(indices.shape[1], dtype=torch.uint8, device=device)

    adj = torch.sparse_coo_tensor(indices, values, size=(N, N), device=device).coalesce()
    return adj


def build_edge_index(coords: np.ndarray, patch_size: int = 512, device: str = "cpu") -> torch.Tensor:

    coords = np.asarray(coords)
    N = coords.shape[0]

    # (x, y) -> node id
    coord_to_index = { (int(x), int(y)): i for i, (x, y) in enumerate(coords) }

    directions = [
        (-patch_size, -patch_size), (0, -patch_size), (patch_size, -patch_size),
        (-patch_size, 0),                              (patch_size, 0),
        (-patch_size, patch_size),  (0, patch_size),  (patch_size, patch_size),
    ]

    src = []
    dst = []

    for i, (x, y) in enumerate(coords):
        x = int(x); y = int(y)
        for dx, dy in directions:
            j = coord_to_index.get((x + dx, y + dy))
            if j is not None:
                src.append(i); dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index

def visualize_patch_grid(coords, adj, wsi, patch_size=512, patch_level=0, max_examples=3, neighbor_n=5):
    neighbor_counts = adj.sum(axis=1)
    # print frequency of neighbor counts
    unique, counts = np.unique(neighbor_counts, return_counts=True)
    print("Neighbor counts frequency:")
    for u, c in zip(unique, counts):
        print(f"  {int(u)} neighbors: {c} patches")
    target_indices = np.where(neighbor_counts == neighbor_n)[0]  # center patch

    # make coordinate to index mapping
    coord_dict = {tuple(coord): idx for idx, coord in enumerate(coords)}

    directions = [
        (-1, -1), ( 0, -1), ( 1, -1),
        (-1,  0), ( 0,  0), ( 1,  0),
        (-1,  1), ( 0,  1), ( 1,  1)
    ]
    shuffle_indices = np.random.permutation(target_indices)
    #for count, center_idx in enumerate(target_indices):
    for count, center_idx in enumerate(shuffle_indices):
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

def _sorted_linear_keys(adj: torch.Tensor):
    """
    Return sorted linear keys (i*N + j) and aligned values (if any) for a sparse COO tensor.
    """
    assert adj.is_sparse, "adj must be a sparse COO tensor"
    adj = adj.coalesce()
    idx = adj.indices()          # [2, nnz]
    N = adj.size(0)
    keys = idx[0] * N + idx[1]   # [nnz]
    order = torch.argsort(keys)
    keys = keys[order]
    vals = adj.values()[order]
    return keys, vals

def is_symmetric_sparse(adj: torch.Tensor, *, rtol=0.0, atol=0.0) -> bool:
    """
    Robust symmetry check for sparse COO.
    - For bool: checks edge existence symmetry.
    - For numeric: checks both structure and value symmetry (with tolerance).
    """
    assert adj.is_sparse, "adj must be a sparse COO tensor"
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be square (N x N)"

    A = adj.coalesce()
    AT = A.transpose(0, 1).coalesce()

    kA, vA = _sorted_linear_keys(A)
    kT, vT = _sorted_linear_keys(AT)

    # same sparsity pattern?
    if kA.numel() != kT.numel():
        return False
    if not torch.equal(kA, kT):
        return False

    # values check
    if A.dtype == torch.bool:
        # existence only
        # (values may not all be True depending on how it was built, so check equality)
        return torch.equal(vA, vT)

    # numeric: allow tolerance if float
    if torch.is_floating_point(vA) or torch.is_complex(vA):
        return torch.allclose(vA, vT, rtol=rtol, atol=atol)
    else:
        return torch.equal(vA, vT)
