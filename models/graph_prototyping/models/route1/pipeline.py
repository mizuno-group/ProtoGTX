#!/usr/bin/env python3
"""
Created on 2025-08-18 (Mon) 22:56:39

1. prototype.pklの作成 ※trainデータのパッチをkmeansでクラスタリング
2. PANTHERの実行によるWSIレベル特徴量の取得
3. PANTHERの実行によるパッチレベル特徴量の取得

@author: I.Azuma
"""
# %% 1. Definition of prototypes
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import os
from os.path import join as j_

import argparse
import torch
from torch.utils.data import DataLoader


from __future__ import print_function

import sys
sys.path.append(BASE_DIR+'/github/PathoGraphX/models/graph_prototyping')
from models.route1.prototyping.proto_dataset import build_proto_datasets
from utils.utils import seed_torch, read_splits
from utils.file_utils import save_pkl
from utils.proto_utils import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
# model / loss fn args ###
parser.add_argument('--model_type', type=str, default='PANTHER')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_proto', type=int, default=16)
parser.add_argument('--n_proto_patches', type=int, default=10000)
parser.add_argument('--n_init', type=int, default=5)
parser.add_argument('--n_iter', type=int, default=50)
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None)
parser.add_argument('--split_dir', type=str, default='/workspace/cluster/HDD/azuma/Pathology_Graph/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info')
parser.add_argument('--split_names', type=str, default='train')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args(args=[])

args.data_source = [BASE_DIR+'/datasource/CPTAC/LSCC_CLAM/patch_512/features/feats_h5',
                    BASE_DIR+'/datasource/CPTAC/LUAD_CLAM/patch_512/features/feats_h5']

train_kwargs = dict(data_source=args.data_source)
seed_torch(args.seed)
csv_splits = read_splits(args)
print('\nsuccessfully read splits for: ', list(csv_splits.keys()))
dataset_splits = build_proto_datasets(csv_splits,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      train_kwargs=train_kwargs)

print('\nInit Datasets...', end=' ')


os.makedirs(j_(args.split_dir, 'prototypes'), exist_ok=True)
loader_train = dataset_splits['train']

_, weights = cluster(loader_train,
                     n_proto=args.n_proto,
                     n_iter=args.n_iter,
                     n_init=args.n_init,
                     feature_dim=args.in_dim,
                     mode=args.mode,
                     n_proto_patches=args.n_proto_patches,
                     use_cuda=True if torch.cuda.is_available() else False)


save_fpath = j_(args.split_dir,
                'prototypes',
                f"prototypes_c{args.n_proto}_{args.data_source[0].split('/')[-2]}_{args.mode}_num_{args.n_proto_patches:.1e}.pkl")

save_pkl(save_fpath, {'prototypes': weights})

# %% 1. PANTHER
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import os
import json
import argparse
import numpy as np
import pandas as pd

from os.path import join as j_

import torch

import sys
sys.path.append(BASE_DIR+'/github/PathoGraphX/models/graph_prototyping')
from utils.utils import (seed_torch, array2list, merge_dict, read_splits)
from utils.file_utils import save_pkl
from models.route1.prototyping.proto_dataset import build_datasets
from models.route1.prototyping.trainer import prot_embed, train

#  Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=100, type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=1)

### model / loss fn args ###
parser.add_argument('--model_type', type=str, choices=['H2T', 'ABMIL', 'TransMIL', 'SumMIL', 'OT', 'PANTHER', 'ProtoCount', 'DeepAttnMIL', 'ILRA'], default='PANTHER',help='type of model')
parser.add_argument('--emb_model_type', type=str, default='LinearEmb')  # NOTE: Changed
parser.add_argument('--ot_eps', default=0.1, type=float, help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_config', type=str, default='PANTHER_cptac', help="name of model config file")
parser.add_argument('--config_dir', type=str, default=BASE_DIR+'/github/PathoGraphX/models/non_param_prototyping/models/configs/',)

parser.add_argument('--in_dim', default=1024, type=int,
                    help='dim of input features')
parser.add_argument('--in_dropout', default=0.0, type=float,
                    help='Probability of dropping out input features.')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)

parser.add_argument('--train_sampler', type=str, default='random', 
                    choices=['random', 'weighted', 'sequential'])
parser.add_argument('--n_fc_layers', type=int)
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str, default='allcat')

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str, default='.')
parser.add_argument('--fix_proto', action='store_true', default=False)
parser.add_argument('--n_proto', type=int)

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--task', type=str, choices=['ebrains_subtyping_coarse', 'ebrains_subtyping_fine', 'panda', 'cptac'], default='cptac')
parser.add_argument('--target_col', type=str, default='label')

# dataset / split args ###
parser.add_argument('--split_dir', type=str, default=BASE_DIR+'/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=True,  # FIXME
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default=BASE_DIR+'/workspace3/new_model_dev/250818_graph_prototyping/route1/results/250818')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')
args = parser.parse_args([])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args.data_source = [BASE_DIR+'/datasource/CPTAC/LSCC_CLAM/patch_512/features/feats_h5',
                    BASE_DIR+'/datasource/CPTAC/LUAD_CLAM/patch_512/features/feats_h5']
args.label_map = {'LUAD': 0, 'LSCC': 1, 'Normal': 2}
args.n_classes = len(set(list(args.label_map.values())))
args.n_proto = 16
args.load_proto = True
args.proto_path = f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info/prototypes/prototypes_c16_features_kmeans_num_1.0e+04.pkl'
args.fix_proto = False


seed_torch(args.seed)
csv_splits = read_splits(args)
sampler_types = {'train': 'sequential', 'val': 'sequential', 'test': 'sequential'}
train_kwargs = dict(data_source=args.data_source,
                        label_map=args.label_map,
                        target_col=args.target_col,
                        bag_size=args.train_bag_size,
                        shuffle=True)

# use the whole bag at test time
val_kwargs = dict(data_source=args.data_source,
                        label_map=args.label_map,
                        target_col=args.target_col,
                        bag_size=args.val_bag_size)


dataset_splits = build_datasets(csv_splits, 
                                model_type=args.model_type,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler_types=sampler_types,
                                train_kwargs=train_kwargs,
                                val_kwargs=val_kwargs)  # (B, n_patch, 1024) ResNet50

# Embedding Phase
datasets = prot_embed(dataset_splits, args)

# %% パッチレベル特徴量の取得
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import h5py
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(BASE_DIR+'/github/PathoGraphX/models/graph_prototyping')
from models.route1.prototyping.model_PANTHER import PANTHER
from models.configs.model_configs import PANTHERConfig


proto_path = f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info/prototypes/prototypes_c16_features_kmeans_num_1.0e+04.pkl'
config_path = f'{BASE_DIR}/github/PathoGraphX/models/graph_prototyping/models/configs/PANTHER_config.json'

model_type = 'PANTHER'
update_dict = {'in_dim': 1024,
               'out_size': 16,
               'load_proto': 1,
               'fix_proto': 1,
               'proto_path': proto_path}
update_dict.update({'n_classes': 3})

# model definition
config = PANTHERConfig.from_pretrained(config_path, update_dict=update_dict)
mode = 'classification'
model = PANTHER(config=config, mode=mode)

# 1. LUAD
l = sorted(glob(BASE_DIR + '/datasource/CPTAC/LUAD_CLAM/patch_512/features/feats_h5/*.h5'))
for path in tqdm(l):
    slide_id = path.split('/')[-1].split('.')[0]
    h5 = h5py.File(path, 'r')
    feats = torch.Tensor(h5['features'][:])
    
    with torch.inference_mode():
        info = model.representation(feats.unsqueeze(dim=0))
        qqs = info['qq']
        #out = info['repr']
        qq = qqs[0,:,:,0].cpu().numpy()
    
    # Save or process `qq` as needed
    save_fpath = f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/results/250822/LUAD/{slide_id}_qq.npy'
    np.save(save_fpath, qq)

# LSCC
l = sorted(glob(BASE_DIR + '/datasource/CPTAC/LSCC_CLAM/patch_512/features/feats_h5/*.h5'))
for path in tqdm(l):
    slide_id = path.split('/')[-1].split('.')[0]
    h5 = h5py.File(path, 'r')
    feats = torch.Tensor(h5['features'][:])
    
    with torch.inference_mode():
        info = model.representation(feats.unsqueeze(dim=0))
        qqs = info['qq']
        #out = info['repr']
        qq = qqs[0,:,:,0].cpu().numpy()
    
    # Save or process `qq` as needed
    save_fpath = f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/results/250822/LSCC/{slide_id}_qq.npy'
    np.save(save_fpath, qq)


