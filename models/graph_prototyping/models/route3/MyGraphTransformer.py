import os
import torch
import numpy as np


import torch.nn as nn
import torch.nn.functional as F

import sys
from .ViT import *
from .gcn import GCNBlock, ProtoGCNBlock

from torch_geometric.nn import dense_mincut_pool
from torch.nn import Linear
class Classifier(nn.Module):
    def __init__(self, n_class, n_features=512, n_proto=16):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        self.proto_dim = 1024

        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim, wo_head=True)  # NOTE: remove head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.CrossEntropyLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(n_features,self.embed_dim,self.bn,self.add_self,self.normalize_embedding,0.,0)       # 64->128
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)                                    # 100-> 20
        self.proto_conv1 = ProtoGCNBlock(in_channel=self.proto_dim, out_channel=self.embed_dim)

        self.head = Linear(n_proto, n_class)


    def forward(self,node_feat,labels,adj,mask,
                proto_features,proto_adj,graphcam_flag=False):
        X=node_feat
        X=mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        s = self.pool1(X)

        if graphcam_flag:
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            os.makedirs('graphcam', exist_ok=True)
            torch.save(s_matrix, path.join('graphcam', 's_matrix.pt'))
            torch.save(s[0], path.join('graphcam', 's_matrix_ori.pt'))
            
            if path.exists(path.join('graphcam', 'att_1.pt')):
                os.remove(path.join('graphcam', 'att_1.pt'))
                os.remove(path.join('graphcam', 'att_2.pt'))
                os.remove(path.join('graphcam', 'att_3.pt'))
    
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)

        out = self.transformer(X)  # (B, 64)

        # Prototype GCN
        proto_out = self.proto_conv1(proto_features, proto_adj)  # (64, n_proto)

        # !! 251007: dot product with prototype features
        dot_c = torch.matmul(out, proto_out)  # (B, n_proto)
        dot_out = self.head(dot_c)  # (B, n_class)

        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        # pred
        pred = dot_out.data.max(1)[1]

        if graphcam_flag:
            print('GraphCAM enabled')
            p = F.softmax(out)
            torch.save(p, path.join('graphcam', 'prob.pt'))

            for index_ in range(p.size(1)):
                one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
                one_hot[0, index_] = out[0][index_]
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * out)       #!!!!!!!!!!!!!!!!!!!!out-->p
                self.transformer.zero_grad()
                one_hot.backward(retain_graph=True)

                kwargs = {"alpha": 1}
                cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device), method="transformer_attribution", is_ablation=False, 
                                            start_layer=0, **kwargs)

                torch.save(cam, path.join('graphcam', 'cam_{}.pt'.format(index_)))

        return pred, labels, loss
