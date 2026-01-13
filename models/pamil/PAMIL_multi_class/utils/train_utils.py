import torch
from enum import Enum
import numpy as np
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.nn as nn
import torch.optim as optim
import collections
from itertools import islice
import math
from dataclasses import dataclass


class TrainMode(Enum):
    WARM = 'warm'
    JOINT = 'joint'
    PUSH = 'push'
    LAST = 'last_only'
    

def _freeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = False

def _unfreeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = True
        
def warm_only(model):
    model.proto.requires_grad = False
    _unfreeze_layer(model.attention_net)
    _unfreeze_layer(model.fc)
    _unfreeze_layer(model.classifiers)
    if model.proto_pred:
        _unfreeze_layer(model.proto_pred_classifier)

def joint(model):
    model.proto.requires_grad = False
    # model.proto.requires_grad = True
    _unfreeze_layer(model.attention_net)
    _unfreeze_layer(model.fc)
    _unfreeze_layer(model.classifiers)
    if model.proto_pred:
        _unfreeze_layer(model.proto_pred_classifier)

def last_only(model):
    model.proto.requires_grad = False
    _unfreeze_layer(model.attention_net)
    _freeze_layer(model.fc)
    _unfreeze_layer(model.classifiers)
    if model.proto_pred:
        _unfreeze_layer(model.proto_pred_classifier)


@dataclass(frozen=True)
class Settings:
    # lr and optimizer settings
    warm_optimizer_lrs:dict
    warm_lr_gamma: float
    
    joint_optimizer_lrs: dict
    joint_lr_step_size: int
    joint_lr_gamma: float
    last_layer_optimizer_lr: dict
    
    coef_crs_ent: float = 1
    coef_clst: float = 0.2
    coef_sep: float = -0.02
    coef_l1: float = 1e-2
    
    # epoch_set
    num_train_epochs=350
    num_last_layer_iterations=30
    push_start: int = 100
    num_warm_epochs:int = 80
    # push_epochs = [160, 190, 220, 240]
    push_epochs = [330]
                   
def get_settings():
    return Settings(
        # joint mode setting
        joint_optimizer_lrs={
            'metric_net': 1e-2,
            'prototype_vectors': 1e-1,
            'attention': 1e-2,
            'last_layer': 1e-2,
        },
        joint_lr_step_size=10,
        joint_lr_gamma=0.8,  # from 
        
        # warm up setting 
        warm_optimizer_lrs={
            'metric_net': 1e-2,
            'prototype_vectors': 1e-1,
            'attention': 1e-2,
            'last_layer': 1e-2,
        },
        warm_lr_gamma=0.95,
        
        # last only setting
        last_layer_optimizer_lr={
            'attention': 1e-2,
            'last_layer': 1e-2,
        },
        )