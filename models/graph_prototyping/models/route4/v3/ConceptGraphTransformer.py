#!/usr/bin/env python3
"""
Created on 2025-10-09 (Thu) 14:01:30

Reference
- https://github.com/IBM/concept_transformer/blob/main/ctc/vit.py


@author: I.Azuma
"""
import os
import numpy as np
from os import path

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import dense_mincut_pool

from .ViT import *
from .gcn import GCNBlock
from .layers import CrossAttention


class Classifier(nn.Module):
    def __init__(self, n_class, n_features=512, expl_w=5.0, graphcam_dir='graphcam'):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        self.graphcam_dir = graphcam_dir

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
        self.expl_w = expl_w  # weight for explanation loss

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
        nn.init.xavier_uniform_(self.concepts)
        self.concept_transformer = CrossAttention(
            dim=self.embed_dim,
            n_outputs=self.embed_dim,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
        )


    def forward(self,node_feat,labels,adj,mask,expl,graphcam_flag=False):
        X=node_feat  # (B, N, D)
        X=mask.unsqueeze(2)*X  # (B, N, D)
        X = self.conv1(X, adj, mask)  # (B, N, embed_dim)

        # concept cross-attention
        X_attn, concept_attn = self.concept_transformer(X, self.concepts)  # (B, N, embed_dim)
        X = X + X_attn  # residual connection
        concept_attn = concept_attn.mean(1)  # average over heads  # (B, N, n_prototypes)
        concept_attn_sum = concept_attn.sum(1)  # (B, n_prototypes)

        del X_attn

        s = self.pool1(X)  # (B, N, node_cluster_num)
        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)

            os.makedirs(self.graphcam_dir, exist_ok=True)
            torch.save(s_matrix, path.join(self.graphcam_dir, 's_matrix.pt'))
            torch.save(s[0], path.join(self.graphcam_dir, 's_matrix_ori.pt'))

            if path.exists(path.join(self.graphcam_dir, 'att_1.pt')):
                os.remove(path.join(self.graphcam_dir, 'att_1.pt'))
                os.remove(path.join(self.graphcam_dir, 'att_2.pt'))
                os.remove(path.join(self.graphcam_dir, 'att_3.pt'))

        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)  # (B, node_cluster_num, embed_dim)

        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)

        # loss1
        loss = self.criterion(out, labels)
        cls_loss = loss + mc1 + o1

        # explanation loss
        expl_loss = concepts_cost(concept_attn_sum, expl)
        expl_loss = self.expl_w * expl_loss
        #loss = cls_loss + expl_loss

        # pred
        pred = out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out)
            torch.save(p, path.join(self.graphcam_dir, 'prob.pt'))

            for index_ in range(p.size(1)):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)       #!!!!!!!!!!!!!!!!!!!!out-->p
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device),
                                               method="transformer_attribution", 
                                               is_ablation=False, 
                                               start_layer=0, **kwargs)

                torch.save(cam, path.join(self.graphcam_dir, 'cam_{}.pt'.format(index_)))

        return pred, labels, cls_loss, expl_loss, concept_attn


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
