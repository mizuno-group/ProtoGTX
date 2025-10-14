#!/usr/bin/env python3
"""
Created on 2025-10-14 (Tue) 13:54:37

@author: I.Azuma
"""
# %%
import h5py
import openslide
import numpy as np

from glob import glob
from tqdm import tqdm

import torch

class VisRelatedPatches():
    def __init__(self, slide_ids, proto_feats, h5_feats_dirs, wsi_dirs,
                 topn=3, patch_size=512, scale=4, proto_norm=False):
        self.slide_ids = [t.split('.')[0] for t in slide_ids]
        self.proto_feats = proto_feats
        tmp_h5_feats_paths = []
        for d in h5_feats_dirs:
            tmp_h5_feats_paths.extend(glob(f'{d}/*.h5'))
        self.h5_feats_paths = []
        for slide_id in self.slide_ids:
            for path in tmp_h5_feats_paths:
                if slide_id in path:
                    self.h5_feats_paths.append(path)
                    break

        tmp_wsi_paths = []
        for d in wsi_dirs:
            tmp_wsi_paths.extend(glob(f'{d}/*'))
        self.wsi_paths = []
        for slide_id in self.slide_ids:
            for path in tmp_wsi_paths:
                if slide_id in path:
                    self.wsi_paths.append(path)
                    break

        assert len(self.h5_feats_paths)==len(self.wsi_paths), f'len(h5)={len(self.h5_feats_paths)}, len(wsi)={len(self.wsi_paths)}'
        self.topn = topn
        self.patch_size = patch_size
        self.scale = scale
        self.proto_norm = proto_norm

    def visualize_related_patches(self, vis=True):
        stack_feats = []
        stack_coords = []
        stack_slide_ids = []
        for slide_id in tqdm(self.slide_ids):
            slide_id = slide_id.split('.')[0]
            h5_feats_fpath = None
            for path in self.h5_feats_paths:
                if slide_id in path:
                    h5_feats_fpath = path
                    break
            
            h5 = h5py.File(h5_feats_fpath,'r')
            feats = h5['features'][:]
            coords = h5['coords'][:]
            h5.close()
            stack_feats.append(feats)
            stack_coords.append(coords)
            stack_slide_ids.extend([slide_id]*feats.shape[0])

        proto_tensor = torch.Tensor(self.proto_feats)
        stack_feats = np.concatenate(stack_feats, axis=0)
        stack_feats = torch.Tensor(stack_feats)
        stack_coords = np.concatenate(stack_coords, axis=0)
        stack_coords = np.array(stack_coords)

        sim = torch.nn.functional.cosine_similarity(
                stack_feats.unsqueeze(1),    
                proto_tensor.unsqueeze(0),
                dim=-1
            )
        self.sim = sim
        if self.proto_norm:
            sim = sim / sim.sum(axis=1, keepdim=True)  # NOTE: normalize similarity

        # top n related patches for each prototype
        topn_related_proto = {}
        for proto_idx in range(proto_tensor.shape[0]):
            sim_scores = sim[:, proto_idx].numpy()
            top_indices = np.argsort(sim_scores)[-self.topn:][::-1]
            topn_related_proto[proto_idx] = top_indices.tolist()
        
        self.topn_related_proto = topn_related_proto

        print(f'Top {self.topn} related prototypes: ', topn_related_proto)

        if vis:
            for p_n, indices in topn_related_proto.items():
                print(f'Prototype {p_n}:')
                for idx in indices:
                    slide_id = stack_slide_ids[idx]  # identify which slide the patch belongs to
                    coord = stack_coords[idx]

                    wsi_path = None
                    for path in self.wsi_paths:
                        if slide_id in path:
                            wsi_path = path
                            break
                    
                    print(f'  Slide ID: {slide_id}, Coord: {coord}, WSI path: {wsi_path}')
                    wsi = openslide.open_slide(wsi_path)
                    patch = wsi.read_region((coord[0], coord[1]), 0, (self.patch_size, self.patch_size)).convert("RGB")
                    display(patch.resize((patch.width//self.scale, patch.height//self.scale)))
                    wsi.close()

if __name__ == '__main__':
    BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

    import random
    import pickle
    import pandas as pd

    import sys
    sys.path.append(BASE_DIR+'/github/PathoGraphX')
    from visualization.proto_vis_utils import VisRelatedPatches

    # load prototype feature
    proto_path = f'{BASE_DIR}/datasource/Mouse_TAA_MDA_240205/250929_splits/prototypes/prototypes_c16_features_kmeans_num_1.0e+04.pkl'
    loader = open(proto_path,'rb')
    file = pickle.load(loader)
    loader.close()
    proto_feats = file['prototypes'].squeeze()

    # patch coverage selection
    info_df = pd.read_csv(f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info/train.csv')
    slide_ids = info_df['slide_id'].tolist()
    random.seed(1)
    random.shuffle(slide_ids)
    slide_ids = slide_ids[:200]  # limit to 200 slides for quick check

    dat = VisRelatedPatches(
        slide_ids,
        proto_feats,
        h5_feats_dirs=[
            f'{BASE_DIR}/datasource/CPTAC/LSCC_CLAM/patch_512/features/feats_h5/',
            f'{BASE_DIR}/datasource/CPTAC/LUAD_CLAM/patch_512/features/feats_h5/'
        ],
        wsi_dirs=[
            '/workspace/HDDX/Pathology_datasource/PKG-CPTAC-LSCC_v10/LSCC/',
            '/workspace/HDDX/Pathology_datasource/PKG-CPTAC-LUAD_v12/LUAD/'
        ],
        topn=3,
        patch_size=512,
        proto_norm=False
    )
    dat.visualize_related_patches(vis=True)

