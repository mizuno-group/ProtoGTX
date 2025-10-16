#!/usr/bin/env python3
"""
Created on 2025-10-14 (Tue) 13:54:37

@author: I.Azuma
"""
# %%
import cv2
import h5py
import openslide
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm

from glob import glob
from tqdm import tqdm
from PIL import Image, ImageOps

import torch

COLORS = [
    '#696969','#556b2f','#a0522d','#483d8b', 
    '#008000','#008b8b','#000080','#7f007f',
    '#8fbc8f','#b03060','#ff0000','#ffa500',
    '#00ff00','#8a2be2','#00ff7f','#FFFF54', 
    '#00ffff','#00bfff','#f4a460','#adff2f',
    '#da70d6','#b0c4de','#ff00ff','#1e90ff',
    '#f0e68c','#0000ff','#dc143c','#90ee90',
    '#ff1493','#7b68ee','#ffefd5','#ffb6c1'
]

def get_mixture_plot(mixtures):
    colors = COLORS
    cmap = {f'c{k}':v for k,v in enumerate(colors[:len(mixtures)])}
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(10,3), dpi=100)

    #prop = fm.FontProperties(fname="./Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mixtures = pd.DataFrame(mixtures, index=cmap.keys()).T
    ax = sns.barplot(mixtures, palette=cmap)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Proportion / Mixture', fontsize=12)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    ax.set_ylim([0, 0.55])
    plt.close()
    return ax.get_figure()

def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = COLORS
    
    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict

def visualize_categorical_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        alpha=0.4,
        verbose=True,
    ):

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        color = label2color_dict[labels[idx][0]]
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img

class VisP2PAttn():
    def __init__(self, wsi_dirs, h5_feats_dirs, patch_size=512, downsample=64, n_prototypes=16):
        self.wsi_dirs = wsi_dirs
        self.h5_feats_dirs = h5_feats_dirs
        self.patch_size = patch_size
        self.downsample = downsample
        self.n_prototypes = n_prototypes

    def visualize_attn(self, slide_id, concept_attn, ind=0):
        slide_id = slide_id.split('.')[0]
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

        wsi = openslide.open_slide(wsi_path)
        coords = h5py.File(h5_feats_fpath, 'r')['coords'][:]

        attn = concept_attn[ind].cpu().detach().numpy()
        global_cluster_labels = attn.argmax(axis=1)
        #count_labels = np.bincount(global_cluster_labels)
        counts = np.bincount(global_cluster_labels, minlength=self.n_prototypes)

        cat_map = visualize_categorical_heatmap(
            wsi,
            coords, 
            global_cluster_labels, 
            label2color_dict=get_default_cmap(self.n_prototypes),
            vis_level=wsi.get_best_level_for_downsample(self.downsample),
            patch_size=(self.patch_size, self.patch_size),
            alpha=0.4,
        )
        display(cat_map.resize((cat_map.width//4, cat_map.height//4)))
        display(get_mixture_plot(mixtures=counts/counts.sum()))


class VisP2P():
    def __init__(self, wsi_dirs, h5_feats_dirs, patch_size=512, downsample=64, n_prototypes=16):
        self.wsi_dirs = wsi_dirs
        self.h5_feats_dirs = h5_feats_dirs
        self.patch_size = patch_size
        self.downsample = downsample
        self.n_prototypes = n_prototypes

    def visualize_initial_mapping(self, slide_id, proto_feats):
        slide_id = slide_id.split('.')[0]
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

        wsi = openslide.open_slide(wsi_path)
        coords = h5py.File(h5_feats_fpath, 'r')['coords'][:]
        feats = h5py.File(h5_feats_fpath, 'r')['features'][:]

        feats_tensor = torch.Tensor(feats)
        proto_tensor = torch.Tensor(proto_feats)

        sim = torch.nn.functional.cosine_similarity(
            feats_tensor.unsqueeze(1),
            proto_tensor.unsqueeze(0),
            dim=-1
        ) 
        global_cluster_labels = torch.argmax(sim, dim=1).numpy()
        counts = np.bincount(global_cluster_labels, minlength=self.n_prototypes)

        cat_map = visualize_categorical_heatmap(
            wsi,
            coords, 
            global_cluster_labels, 
            label2color_dict=get_default_cmap(self.n_prototypes),
            vis_level=wsi.get_best_level_for_downsample(self.downsample),
            patch_size=(self.patch_size, self.patch_size),
            alpha=0.4,
        )
        display(cat_map.resize((cat_map.width//4, cat_map.height//4)))
        display(get_mixture_plot(mixtures=counts/counts.sum()))

        return global_cluster_labels


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

