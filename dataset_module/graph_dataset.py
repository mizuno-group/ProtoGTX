# -*- coding: utf-8 -*-
"""
Created on 2025-06-11 (Wed) 18:26:07

Reference
- https://github.com/vkola-lab/tmi2022/blob/main/utils/dataset.py

@author: I.Azuma
"""
# %%
"""Dataset class for the graph classification task."""

import os
import numpy as np
from typing import Any

import torch
from torch.utils import data
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "site/graph_name\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)

        The dataset directory structure should be as follows:
        root/
        └── {site}_features/
            └── simclr_files/
                ├── graph1/
                │   ├── features.pt
                │   └── adj_s.pt
                ├── graph2/
                │   ├── features.pt
                │   └── adj_s.pt
                └── ...
        """
        super(GraphDataset, self).__init__()
        self.root = root
        self.ids = ids
        self.classdict = classdict
        self.site = site


    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split the id into graph_name and label
            graph_name = info.split('\t')[0]
            if '/' in graph_name:
                site, graph_file = graph_name.split('/')
            else:
                site = 'unknown'
                graph_file = graph_name
            label = info.split('\t')[1]
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc


        sample: dict[str, Any] = {}
        sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample['id'] = graph_file

        feature_path = os.path.join(self.root, graph_file, f'features_{graph_file}.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'features.pt for {graph_file} doesn\'t exist')

        adj_s_path = os.path.join(self.root, graph_file, f'adj_s_{graph_file}.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_file} doesn\'t exist')

        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        sample['image'] = features
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)

class ConcatGraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 root2: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "site/graph_name\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)

        The dataset directory structure should be as follows:
        root/
        └── {site}_features/
            └── simclr_files/
                ├── graph1/
                │   ├── features.pt
                │   └── adj_s.pt
                ├── graph2/
                │   ├── features.pt
                │   └── adj_s.pt
                └── ...
        root2/
        ├── graph1_features.pt
        ├── graph2_features.pt
        └── ...
        """
        super(ConcatGraphDataset, self).__init__()
        self.root = root
        self.root2 = root2
        self.ids = ids
        self.classdict = classdict
        self.site = site


    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split the id into graph_name and label
            graph_name = info.split('\t')[0]
            if '/' in graph_name:
                site, graph_file = graph_name.split('/')
            else:
                site = 'unknown'
                graph_file = graph_name
            label = info.split('\t')[1]
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc


        sample: dict[str, Any] = {}
        sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample['id'] = graph_file
        
        # Concat features
        feature_path = os.path.join(self.root2, f'{graph_file}_concat.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'features.pt for {graph_file} doesn\'t exist')


        adj_s_path = os.path.join(self.root, graph_file, f'adj_s_{graph_file}.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_file} doesn\'t exist')

        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        # concatenate
        sample['image'] = features
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)

class ConceptGraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 root2: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "site/graph_name\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)

        The dataset directory structure should be as follows:
        root/
        └── {site}_features/
            └── simclr_files/
                ├── graph1/
                │   ├── features.pt
                │   └── adj_s.pt
                ├── graph2/
                │   ├── features.pt
                │   └── adj_s.pt
                └── ...
        root2/
        ├── id1_expl.pt
        ├── id2_expl.pt
        └── ...
        """
        super(ConceptGraphDataset, self).__init__()
        self.root = root
        self.root2 = root2
        self.ids = ids
        self.classdict = classdict
        self.site = site


    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split the id into graph_name and label
            graph_name = info.split('\t')[0]
            if '/' in graph_name:
                site, graph_file = graph_name.split('/')
            else:
                site = 'unknown'
                graph_file = graph_name
            label = info.split('\t')[1]
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc


        sample: dict[str, Any] = {}
        sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample['id'] = graph_file
        
        # 1. patch features
        feature_path = os.path.join(self.root, graph_file, f'features_{graph_file}.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'features.pt for {graph_file} doesn\'t exist')
        
        # 2. expl tensor
        expl_path = os.path.join(self.root2, f'{graph_file}_expl.pt')
        if os.path.exists(expl_path):
            expl = torch.load(expl_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'expl.pt for {graph_file} doesn\'t exist')

        adj_s_path = os.path.join(self.root, graph_file, f'adj_s_{graph_file}.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_file} doesn\'t exist')

        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        sample['image'] = features
        sample['expl'] = expl
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)

class DualConceptGraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self,
                 root: str,
                 l_expl_root: str,
                 g_expl_root: str,
                 ids: list[str],
                 site: str | None = 'LUAD',
                 classdict: dict[str, int] | None = None,
                 ) -> None:
        """Create a GraphDataset.

        Args:
            root (str): Path to the dataset's root directory.
            ids (list[str]): List of ids of the images to load.
                Each id should be a string in the format "site/graph_name\tlabel".
            site (str | None): Name of the canonical tissue site the images from. The only sites
                that are recognized as canonical (i.e., they have a pre-defined classdict) are
                'LUAD', 'LSCC', 'NLST', and 'TCGA'. If your dataset is not a canonical site, leave
                this as None. 
            classdict (dict[str, int]): Dictionary mapping the class names to the class indices. Not
                needed if your dataset is a canonical site or your labels are already 0-indexed
                positive consecutive integers.
            target_patch_size (int | None): Size of the patches to extract from the images. (Not
                used.)

        The dataset directory structure should be as follows:
        root/
        └── {site}_features/
            └── simclr_files/
                ├── graph1/
                │   ├── features.pt
                │   └── adj_s.pt
                ├── graph2/
                │   ├── features.pt
                │   └── adj_s.pt
                └── ...
        l_expl_root/
        ├── id1_expl.pt
        ├── id2_expl.pt
        └── ...
        g_expl_root/
        ├── id1_expl.pt
        ├── id2_expl.pt
        └── ...
        """
        super(DualConceptGraphDataset, self).__init__()
        self.root = root
        self.l_expl_root = l_expl_root
        self.g_expl_root = g_expl_root
        self.ids = ids
        self.classdict = classdict
        self.site = site


    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.ids[index].replace('\n', '')
        try:
            # Split the id into graph_name and label
            graph_name = info.split('\t')[0]
            if '/' in graph_name:
                site, graph_file = graph_name.split('/')
            else:
                site = 'unknown'
                graph_file = graph_name
            label = info.split('\t')[1]
        except ValueError as exc:
            raise ValueError(
                f"Invalid id format: {info}. Expected format is 'site/filename\tlabel'") from exc


        sample: dict[str, Any] = {}
        sample['label'] = self.classdict[label] if (self.classdict is not None) else int(label)
        sample['id'] = graph_file
        
        # 1. patch features
        feature_path = os.path.join(self.root, graph_file, f'features_{graph_file}.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'features.pt for {graph_file} doesn\'t exist')
        
        # 2. local expl tensor
        l_expl_path = os.path.join(self.l_expl_root, f'{graph_file}_expl.pt')
        if os.path.exists(l_expl_path):
            l_expl = torch.load(l_expl_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'Local expl.pt for {graph_file} doesn\'t exist')
        
        # 3. global expl tensor
        g_expl_path = os.path.join(self.g_expl_root, f'{graph_file}_expl.pt')
        if os.path.exists(g_expl_path):
            g_expl = torch.load(g_expl_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'Global expl.pt for {graph_file} doesn\'t exist')

        adj_s_path = os.path.join(self.root, graph_file, f'adj_s_{graph_file}.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location='cpu')
        else:
            raise FileNotFoundError(f'adj_s.pt for {graph_file} doesn\'t exist')

        if adj_s.is_sparse:
            adj_s = adj_s.to_dense()

        sample['image'] = features
        sample['l_expl'] = l_expl
        sample['g_expl'] = g_expl
        sample['adj_s'] = adj_s

        return sample

    def __len__(self):
        return len(self.ids)

