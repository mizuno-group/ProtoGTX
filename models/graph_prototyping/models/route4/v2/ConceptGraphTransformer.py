#!/usr/bin/env python3
"""
Created on 2025-10-09 (Thu) 14:01:30

Reference
- https://github.com/IBM/concept_transformer/blob/main/ctc/vit.py


@author: I.Azuma
"""

import sys
import os
import torch
import random
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ViT import *
from .gcn import GCNBlock
from .layers import CrossAttention

from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear


class Classifier(nn.Module):
    def __init__(self, n_class, n_features=512, expl_w=5.0):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(n_features,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0)
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)

        ### concept parameters ###
        self.n_local_concepts = 16
        self.n_global_concepts = 16
        self.num_heads = 2
        self.attention_dropout = 0.1
        self.projection_dropout = 0.1
        self.expl_w = expl_w  # weight for explanation loss

        # local
        self.local_concepts = nn.Parameter(
            torch.zeros(1, self.n_local_concepts, self.embed_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.local_concepts)
        self.local_concept_transformer = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
        )

        # global
        self.global_concepts = nn.Parameter(
            torch.zeros(1, self.n_global_concepts, self.embed_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.global_concepts)

        self.global_concept_transformer = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
        )


    def forward(self,node_feat,labels,adj,mask,local_expl,global_expl):
        X=node_feat  # (B, N, D)
        X=mask.unsqueeze(2)*X  # (B, N, D)
        X = self.conv1(X, adj, mask)  # (B, N, embed_dim)

        # local concept transformer
        X_attn1, local_concept_attn = self.local_concept_transformer(X, self.local_concepts)  # (B, N, embed_dim)
        X = X + X_attn1  # residual connection
        local_concept_attn = local_concept_attn.mean(1)  # average over heads  # (B, N, n_prototypes)
        local_concept_attn_sum = local_concept_attn.sum(1)  # (B, n_prototypes)

        del X_attn1

        s = self.pool1(X)  # (B, N, node_cluster_num)
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)  # (B, node_cluster_num, embed_dim)

        # global concept transformer
        X_attn2, global_concept_attn = self.global_concept_transformer(X, self.global_concepts)  # (B, N, embed_dim)
        X = X + X_attn2  # residual connection
        global_concept_attn = global_concept_attn.mean(1)  # average over heads  # (B, N, n_prototypes)
        global_concept_attn_sum = global_concept_attn.sum(1)  # (B, n_prototypes)

        del X_attn2

        out = self.transformer(X)

        # loss1
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1

        # explanation loss
        expl_loss = concepts_cost(local_concept_attn_sum, local_expl) \
            + concepts_cost(global_concept_attn_sum, global_expl)
        loss = loss + self.expl_w * expl_loss  # expl_lambda=5.0

        # pred
        pred = out.data.max(1)[1]

        return pred,labels,loss,(local_concept_attn,global_concept_attn)


# concept loss functions
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()

def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost

def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    #if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    norm_concept_attn = concept_attn[idx] / concept_attn[idx].sum(-1, keepdims=True)
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(norm_concept_attn, norm_attn_targets, reduction="mean")
