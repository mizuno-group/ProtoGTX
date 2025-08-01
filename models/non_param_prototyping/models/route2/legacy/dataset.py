# -*- coding: utf-8 -*-
"""
Created on 2025-07-31 (Thu) 14:47:56

proto-non-param内のdinov2の挙動の確認

@author: I.Azuma
"""
# %%
import h5py
import openslide
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as T

class WSIDataset():
    def __init__(self, file_path, wsi, img_transforms=None):
        """
        Args:
            file_path (str): Path to the .h5 file containing patched data.
            wsi: OpenSlide object or similar, used to extract patches from WSI.
            img_transforms (callable, optional): Transformations to apply on the extracted image patch.
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.h5 = h5py.File(self.file_path, "r")  # ← keep open

        self.coords = self.h5['coords']
        self.patch_level = self.coords.attrs['patch_level']
        self.patch_size = self.coords.attrs['patch_size']
        self.length = len(self.coords)
            
        self.summary()
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.roi_transforms is not None:
            img = self.roi_transforms(img)
        
        if isinstance(img, Image.Image):
            img = ToTensor()(img)

        return {'img': img, 'coord': coord}, idx
    
    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('transformations: ', self.roi_transforms)

# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Pathology_Graph'

from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import einsum, rearrange, repeat

import sys
sys.path.append(f"{BASE_DIR}/github/PathoGraphX/models/non_param_prototyping")
from models.route2.backbone import DINOv2BackboneExpanded

file_name = "/C3L-00081-21"
patch_path = f"{BASE_DIR}/datasource/CPTAC/LSCC_CLAM/patch_512/patches/{file_name}.h5"
feature_path = f"{BASE_DIR}/datasource/CPTAC/LSCC_CLAM/patch_512/features/h5_files/{file_name}.h5"
wsi_path = f"/workspace/mnt/HDDX/Pathology_datasource/PKG-CPTAC-LSCC_v10/LSCC/{file_name}.svs"

normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
transforms = T.Compose([
    T.Resize((224, 224,)),
    T.ToTensor(),
    normalize
])
wsi = openslide.open_slide(wsi_path)
dataset = WSIDataset(file_path=patch_path, wsi=wsi, img_transforms=transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# load backbone
backbone = DINOv2BackboneExpanded(
                name="dinov2_vitb14",
                n_splits=1,
                mode="block_expansion",
                freeze_norm_layer=True
            )
backbone = backbone.to("cuda")

for tmp, _ in tqdm(dataloader):
    x = tmp['img'].to("cuda")
    patch_tokens, raw_patch_tokens, cls_tokens = backbone(x)
    
# Forward pass through the backbone
prototypes = torch.randn(3+1, 5, 768).to("cuda")  # 768 is the feature dimension (dinov2_vitb14)
patch_tokens, raw_patch_tokens, cls_tokens = backbone(x)

patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
prototype_norm = F.normalize(prototypes, p=2, dim=-1)

patch_prototype_logits = einsum(patch_tokens, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
image_prototype_logits = patch_prototype_logits.max(1).values  # shape: [B, C, K,], C=n_classes+1
