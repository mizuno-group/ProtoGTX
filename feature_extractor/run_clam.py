# -*- coding: utf-8 -*-
"""
Created on 2025-06-04 (Wed) 22:06:27

Extract features with CLAM

CLAM: https://github.com/mahmoodlab/CLAM

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/Pathology_Graph'

import os
os.chdir(BASE_DIR+'/github/CLAM')

# %%
### 1. Patch Extraction and Segmentation with CLAM
# LUAD
! python create_patches_fp.py --source /workspace/mnt/HDDX/Pathology_datasource/PKG-CPTAC-LUAD_v12/LUAD --save_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LUAD_CLAM --patch_size 256 --seg --patch --stitch

# LSCC
! python create_patches_fp.py --source /workspace/mnt/HDDX/Pathology_datasource/PKG-CPTAC-LSCC_v10/LSCC --save_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LSCC_CLAM --patch_size 256 --seg --patch --stitch 

# %%
### 2. Feature Extraction with CLAM
import glob
import pandas as pd

state = "LUAD"

df = pd.read_csv(BASE_DIR+f'/datasource/CPTAC/{state}_CLAM/process_list_autogen.csv')

processed_names = sorted(glob.glob(BASE_DIR+f'/datasource/CPTAC/{state}_CLAM/patches/*.h5'))
processed_names = [t.split('.')[0].split('/')[-1]+".svs" for t in processed_names]

edited_df = df[df['slide_id'].isin(processed_names)].reset_index(drop=True)
#edited_df.to_csv(BASE_DIR+f'/datasource/CPTAC/{state}_CLAM/process_list_edited.csv')

! CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LUAD_CLAM --data_slide_dir /workspace/mnt/HDDX/Pathology_datasource/PKG-CPTAC-LUAD_v12/LUAD  --csv_path /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LUAD_CLAM/process_list_edited.csv --feat_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LUAD_CLAM/features --batch_size 32 --slide_ext .svs

! CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LSCC_CLAM --data_slide_dir /workspace/mnt/HDDX/Pathology_datasource/PKG-CPTAC-LSCC_v10/LSCC  --csv_path /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LSCC_CLAM/process_list_edited.csv --feat_dir /workspace/mnt/cluster/HDD/azuma/Pathology_Graph/datasource/CPTAC/LSCC_CLAM/features --batch_size 32 --slide_ext .svs

# %%
### 3. Train, Valid, Test Split
import random

# 1. Preparation of meta information
luad_path = BASE_DIR+'/datasource/CPTAC/meta_info/CPTAC_LUAD_meta.json'
df1 = pd.read_json(luad_path, orient='records')
# rename Specimen_Type
df1['label'] = df1['Specimen_Type'].replace({'tumor_tissue': 'LUAD', 'normal_tissue': 'Normal'})

lscc_path = BASE_DIR+'/datasource/CPTAC/meta_info/CPTAC_LSCC_meta.json'
df2 = pd.read_json(lscc_path, orient='records')

# rename Specimen_Type
df2['label'] = df2['Specimen_Type'].replace({'tumor_tissue': 'LSCC', 'normal_tissue': 'Normal'})
df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

#df.to_csv(BASE_DIR+'/github/PANTHER/src/splits/classification/CPTAC/meta_info.csv', index=False)

# 2. Train Valid Test Split
df_luad = pd.read_csv(BASE_DIR+'/datasource/CPTAC/LUAD_CLAM/process_list_edited.csv',index_col=0)
df_lscc = pd.read_csv(BASE_DIR+'/datasource/CPTAC/LSCC_CLAM/process_list_edited.csv',index_col=0)
concat_df = pd.concat([df_luad, df_lscc], axis=0).reset_index(drop=True)

label_df = df[['Slide_ID', 'label']]
label_df['slide_id'] = label_df['Slide_ID'].apply(lambda x: x+'.svs') 

# join with slide_id
concat_df = concat_df.merge(label_df, how='left', on='slide_id')
concat_df = concat_df[concat_df['label'].isin(['LUAD', 'LSCC', 'Normal'])].reset_index(drop=True)

# train:val:test = 8:1:1
concat_df = concat_df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame

train_df = concat_df[:int(len(concat_df)*0.8)]
val_df = concat_df[int(len(concat_df)*0.8):int(len(concat_df)*0.9)]
test_df = concat_df[int(len(concat_df)*0.9):]

#train_df.to_csv(BASE_DIR+'/github/PANTHER/src/splits/classification/CPTAC/train.csv', index=False)
#val_df.to_csv(BASE_DIR+'/github/PANTHER/src/splits/classification/CPTAC/val.csv', index=False)
#test_df.to_csv(BASE_DIR+'/github/PANTHER/src/splits/classification/CPTAC/test.csv', index=False)
