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
    def __init__(self, n_class, n_features: int = 512):
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
        self.n_concepts = 16
        self.n_unsup_concepts = 16
        self.num_heads = 2
        self.attention_dropout = 0.1
        self.projection_dropout = 0.1

        # unsupervised
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, self.n_unsup_concepts, self.embed_dim), requires_grad=True
        )
        self.unsup_concept_tranformer = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
        )

        # supervised
        self.concepts = nn.Parameter(
            torch.zeros(1, self.n_concepts, self.embed_dim), requires_grad=True
        )
        self.concept_transformer = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
        )


    def forward(self,node_feat,labels,adj,mask,expl):
        X=node_feat  # (B, N, D)
        X=mask.unsqueeze(2)*X  # (B, N, D)
        X = self.conv1(X, adj, mask)  # (B, N, embed_dim)

        # concept cross-attention
        X, concept_attn = self.concept_transformer(X, self.concepts)  # (B, N, embed_dim)
        concept_attn = concept_attn.mean(1)  # average over heads  # (B, N, n_prototypes)
        concept_attn_sum = concept_attn.sum(1)  # (B, n_prototypes)

        s = self.pool1(X)  # (B, N, node_cluster_num)
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)  # (B, node_cluster_num, embed_dim)

        out = self.transformer(X)

        # loss1
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1

        # explanation loss
        expl_loss = concepts_cost(concept_attn_sum, expl)
        loss = loss + 1.0 * expl_loss  # expl_lambda=5.0

        # pred
        pred = out.data.max(1)[1]

        return pred,labels,loss,concept_attn


class ConceptTransformerVIT(nn.Module):
    """Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end.
    The difference with the version in ctc.py is that instead of using sequence pooling for global concepts it
    uses the embedding of the cls token of the VIT
    """
    def __init__(
        self,
        embedding_dim=768,
        num_classes=10,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
        n_unsup_concepts=10,
        n_concepts=10,
        n_spatial_concepts=10,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_unsup_concepts > 0:
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Non-spatial concepts
        self.n_concepts = n_concepts
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_concepts > 0:
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x_cls, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0.0
        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_cls, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_cls, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn

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
