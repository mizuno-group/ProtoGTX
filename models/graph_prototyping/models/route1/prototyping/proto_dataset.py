#!/usr/bin/env python3
"""
Created on 2025-08-18 (Mon) 20:48:35

@author: I.Azuma
"""
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader, sampler

sys.path.append(BASE_DIR+'/github/PathoGraphX/models/graph_prototyping')
from wsi_datasets import WSIClassificationDataset, WSIProtoDataset

def build_sampler(dataset, sampler_type=None):
    data_sampler = None
    if sampler_type is None:
        return data_sampler
    
    assert sampler_type in ['weighted', 'random', 'sequential']
    if sampler_type == 'weighted':
        labels = dataset.get_labels(np.arange(len(dataset)), apply_transform=True)
        uniques, counts = np.unique(labels, return_counts=True)
        weights = {uniques[i]: 1. / counts[i] for i in range(len(uniques))}
        samples_weight = np.array([weights[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        data_sampler = sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    elif sampler_type == 'random':
        data_sampler = sampler.RandomSampler(dataset)
    elif sampler_type == 'sequential':
        data_sampler = sampler.SequentialSampler(dataset)

    return data_sampler

def build_clf_datasets(csv_splits, batch_size=1, num_workers=1,
                   train_kwargs={}, val_kwargs={}, sampler_types={'train': 'random',
                                                                  'val': 'sequential',
                                                                  'test': 'sequential'}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        print("\nSPLIT: ", k)
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        if k == 'test_nlst':
            dataset_kwargs['sample_col'] = 'case_id'
        dataset = WSIClassificationDataset(df, **dataset_kwargs)
        data_sampler = build_sampler(dataset, sampler_type=sampler_types.get(k, 'sequential'))

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
    return dataset_splits

def build_proto_datasets(csv_splits, batch_size=1, num_workers=1, train_kwargs={}):
    dataset_splits = {}
    for k in csv_splits.keys(): # ['train']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy()
        dataset = WSIProtoDataset(df, **dataset_kwargs)

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')

    return dataset_splits