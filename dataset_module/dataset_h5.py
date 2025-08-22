# -*- coding: utf-8 -*-
"""
Created on 2025-06-05 (Thu) 18:38:11

@author: I.Azuma
"""
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Pathology_Graph'

import h5py
import openslide

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import rgb_to_grayscale

class WSI_Patch_Bag(Dataset):
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


def compute_bkg_mask(batch_tensor, threshold=220/255.0):
    """
    Computes a binary mask where pixels above the grayscale threshold are marked as background (1).

    Args:
        batch_tensor (Tensor): [B, 3, H, W] RGB images, values in [0, 1].
        threshold (float): Grayscale threshold (default: 220/255).

    Returns:
        Tensor: [B, H, W] mask with background = 1, foreground = 0.
    """
    gray = 0.299 * batch_tensor[:, 0] + 0.587 * batch_tensor[:, 1] + 0.114 * batch_tensor[:, 2]
    return (gray >= threshold).float()

def qc_background(batch_tensor, max_bg_ratio=0.5):
    """
    Filters images with background ratio below the given threshold.

    Args:
        batch_tensor (Tensor): [B, 3, H, W] images.
        max_bg_ratio (float): Max allowed background ratio per image.

    Returns:
        passed_idx (Tensor): Indices of images that pass the QC.
    """
    bg_mask = compute_bkg_mask(batch_tensor)
    bg_ratio = bg_mask.mean(dim=(1, 2))
    passed_idx = (bg_ratio < max_bg_ratio).nonzero(as_tuple=True)[0]

    return passed_idx, bg_ratio