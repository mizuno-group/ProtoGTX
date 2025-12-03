#!/usr/bin/env python3
"""
Created on 2025-08-20 (Wed) 14:26:11

■ load_proto=True, fix_proto=True
metrics = {
    "acc_val": 0.8894230769230769,
    "bacc_val": 0.8863702254113214,
    "kappa_val": 0.8609988264441257,
    "roc_auc_val": 0.9787997390737116,
    "weighted_f1_val": 0.8885953884151376,
    "bag_size_val": 32784.0,
    "cls_acc_val": 0.8894230769230769,
    "cls_loss_val": 0.27224883729174437,
    "loss_val": 0.27224883729174437,
    "acc_test": 0.8899521531100478,
    "bacc_test": 0.8882255389718076,
    "kappa_test": 0.8570284474847751,
    "roc_auc_test": 0.9794599143585555,
    "weighted_f1_test": 0.8902631446968052,
    "bag_size_test": 32784.0,
    "cls_acc_test": 0.8899521531100478,
    "cls_loss_test": 0.2793099699120269,
    "loss_test": 0.2793099699120269
}

■ load_proto=False, fix_proto=True
metrics_random = {
    "acc_val": 0.9086538461538461,
    "bacc_val": 0.9048253243458723,
    "kappa_val": 0.895616136760992,
    "roc_auc_val": 0.9815952340685978,
    "weighted_f1_val": 0.90775140825561,
    "bag_size_val": 32784.0,
    "cls_acc_val": 0.9086538461538461,
    "cls_loss_val": 0.2586609203646311,
    "loss_val": 0.2586609203646311,
    "acc_test": 0.8947368421052632,
    "bacc_test": 0.8926699834162521,
    "kappa_test": 0.8809914587171328,
    "roc_auc_test": 0.9785356798099057,
    "weighted_f1_test": 0.8949673475015825,
    "bag_size_test": 32784.0,
    "cls_acc_test": 0.8947368421052632,
    "cls_loss_test": 0.2765580933053953,
    "loss_test": 0.2765580933053953
}

@author: I.Azuma
"""
# %%
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
from models.route1.prototyping.proto_dataset import build_clf_datasets
from models.route1.prototyping.trainer import train, prot_embed


#  Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str,
                    choices=['adamW', 'sgd'], default='adamW')
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=1)

### Earlystopper args ###
parser.add_argument('--early_stopping', action='store_true',
                    default=False, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=15,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=10,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

##
# model / loss fn args ###
parser.add_argument('--model_type', type=str, choices=['H2T', 'ABMIL', 'TransMIL', 'SumMIL', 'OT', 'PANTHER', 'ProtoCount', 'DeepAttnMIL', 'ILRA'], default='PANTHER',help='type of model')
parser.add_argument('--emb_model_type', type=str, default='LinearEmb')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
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
parser.add_argument('--exp_code', type=str,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, choices=['ebrains_subtyping_coarse', 'ebrains_subtyping_fine', 'panda', 'cptac'], default='cptac')
parser.add_argument('--target_col', type=str, default='label')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
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


dataset_splits = build_clf_datasets(csv_splits, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    sampler_types=sampler_types,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)  # (B, n_patch, 1024) ResNet50

# %%
# Training Phase
fold_results, fold_dumps = train(datasets=dataset_splits, args=args, mode='classification')

all_results, all_dumps = {}, {}
for split, split_results in fold_results.items():
    all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
    save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl

final_dict = {}
for split, split_results in all_results.items():
    final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
final_df = pd.DataFrame(final_dict)
save_name = 'summary.csv'
final_df.to_csv(j_(args.results_dir, save_name), index=False)
with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
    f.write(json.dumps(final_dict, sort_keys=True, indent=4))

dump_path = j_(args.results_dir, 'all_dumps.h5')
fold_dumps.update({'labels': np.array(list(args.label_map.keys()), dtype=np.object_)})
save_pkl(dump_path, fold_dumps)

# %%
