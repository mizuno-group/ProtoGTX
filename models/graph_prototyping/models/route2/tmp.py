#!/usr/bin/env python3
"""
Created on 2025-08-29 (Fri) 17:26:36

@author: I.Azuma
"""
#!/usr/bin/env python3
"""
Created on 2025-08-29 (Fri) 15:20:44

route2の初期開発コード

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import argparse
import pandas as pd

import sys
sys.path.append(BASE_DIR+'/github/PathoGraphX/models/graph_prototyping')

from models.route2.proto_dataset import build_clf_datasets
from utils.utils import seed_torch, read_splits

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
# model / loss fn args ###
parser.add_argument('--model_type', type=str, default='PANTHER')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_proto', type=int, default=16)
parser.add_argument('--n_proto_patches', type=int, default=10000)
parser.add_argument('--n_init', type=int, default=5)
parser.add_argument('--n_iter', type=int, default=50)
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None)
parser.add_argument('--split_dir', type=str, default='/workspace/cluster/HDD/azuma/Pathology_Graph/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info')
parser.add_argument('--split_names', type=str, default='train,val,test')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
args = parser.parse_args(args=[])

args.data_source = [BASE_DIR+'/datasource/CPTAC/LSCC_CLAM/patch_512/features/feats_h5',
                    BASE_DIR+'/datasource/CPTAC/LUAD_CLAM/patch_512/features/feats_h5']
label_map = {'LUAD': 0, 'LSCC': 1, 'Normal': 2}
sampler_types = {'train': 'sequential', 'val': 'sequential', 'test': 'sequential'}

train_kwargs = dict(data_source=args.data_source,
                        label_map=label_map,
                        target_col='label',
                        bag_size=args.train_bag_size,
                        shuffle=True)
val_kwargs = dict(data_source=args.data_source,
                        label_map=label_map,
                        target_col='label',
                        bag_size=args.val_bag_size)
seed_torch(args.seed)
csv_splits = read_splits(args)

dataset_splits = build_clf_datasets(csv_splits, 
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler_types=sampler_types,
                                train_kwargs=train_kwargs,
                                val_kwargs=val_kwargs)

train_loader = dataset_splits['train']
val_loader = dataset_splits['val']
test_loader = dataset_splits['test']

tmp = next(iter(train_loader))

# load prototype checkpoints (Panther-derived)
proto_path = f'{BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route1/split_info/prototypes/prototypes_c16_features_kmeans_num_1.0e+04.pkl'
proto = pd.read_pickle(proto_path)['prototypes'][0,::]

# %%
from __future__ import annotations
import math, random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 0) Loader adapter: convert your batch(dict) → (feats_list, labels, metas)
# =========================================================
def _to_feats_list_from_batch(batch: Dict, in_dim: int) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict]]:
    """
    Accepts a batch from your DataLoader:
        batch = {'img': Tensor, 'coords': Tensor(optional), 'label': Tensor}
    Returns:
        feats_list: List[Tensor(M_i, in_dim)]
        labels:    LongTensor(B,)
        metas:     List[dict] (store coords if provided)
    """
    imgs = batch["img"]  # could be (B, M, D) or (B, D, M) or (M, D)
    labels = batch["label"]
    coords = batch.get("coords", None)

    if imgs.dim() == 2:
        # (M, D)
        B = 1
        imgs = imgs.unsqueeze(0)  # -> (1, M, D) or (1, D, M) depending on axis
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if coords is not None and coords.dim() == 2:
            coords = coords.unsqueeze(0)

    B = imgs.size(0)

    # unify shape to (B, M, D)
    if imgs.size(-1) == in_dim:
        # already (B, M, D)
        imgs_bmd = imgs
    elif imgs.size(1) == in_dim:
        # (B, D, M) -> (B, M, D)
        imgs_bmd = imgs.transpose(1, 2)
    else:
        raise ValueError(f"`img` last or second dim must be {in_dim}, but got {tuple(imgs.shape)}")

    feats_list = [imgs_bmd[b] for b in range(B)]  # each: (M, in_dim)
    labels = labels.long().view(-1)

    metas: List[Dict] = []
    for b in range(B):
        meta_b = {}
        if coords is not None:
            if coords.dim() == 3:
                meta_b["coords"] = coords[b]
            elif coords.dim() == 2:
                meta_b["coords"] = coords
        metas.append(meta_b)
    return feats_list, labels, metas


# =========================================================
# 1) Model (same as before, unchanged API)
# =========================================================
class PPCAC(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        proj_dim: int = 1024,
        num_prototypes: int = 128,
        num_classes: int = 2,
        tau_init: float = 0.07,
        pool: str = "mean",
        proto_init: Optional[torch.Tensor] = None,  # (N, proj_dim)
        proto_learnable: bool = False,
        classifier_mode: str = "linear",            # 'linear' or 'class_query'
    ):
        super().__init__()
        assert classifier_mode in ("linear", "class_query")
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.N = num_prototypes
        self.C = num_classes
        self.pool = pool
        self.classifier_mode = classifier_mode

        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))

        if proto_init is not None:
            assert proto_init.shape == (num_prototypes, proj_dim)
            P = F.normalize(proto_init, dim=1)
        else:
            P = F.normalize(torch.randn(num_prototypes, proj_dim), dim=1)
        self.prototypes = nn.Parameter(P, requires_grad=proto_learnable)

        if classifier_mode == "linear":
            self.cls = nn.Linear(num_prototypes, num_classes, bias=True)
        else:
            self.class_queries = nn.Parameter(F.normalize(torch.randn(num_classes, proj_dim), dim=1))
            self.cls_bias = nn.Parameter(torch.zeros(num_classes))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def forward_single(self, x: torch.Tensor, return_attn: bool = True) -> Dict[str, torch.Tensor]:
        # x: (M, in_dim)
        z = self._norm(self.proj(x))             # (M,d)
        P = self._norm(self.prototypes)          # (N,d)
        tau = torch.exp(self.log_tau).clamp_min(1e-4)

        sims = z @ P.t()                         # (M,N)
        A = F.softmax(tau * sims, dim=1)         # (M,N)

        s = A.sum(dim=0) if self.pool == "sum" else A.mean(dim=0)

        if self.classifier_mode == "linear":
            logits = self.cls(s)                 # (C,)
            B = F.softmax(self.cls.weight, dim=1)  # (C,N)
        else:
            Q = self._norm(self.class_queries)   # (C,d)
            B = F.softmax((Q @ P.t()) / math.sqrt(self.proj_dim), dim=1)  # (C,N)
            logits = (B @ s) + self.cls_bias

        out = {"logits": logits, "s": s}
        if return_attn:
            out["A"] = A
            out["B"] = B
        return out

    def forward(self, feats_list: List[torch.Tensor], return_attn: bool = False) -> Dict[str, torch.Tensor]:
        logits_list, s_list, A_list, B_list = [], [], [], []
        for x in feats_list:
            o = self.forward_single(x, return_attn=return_attn)
            logits_list.append(o["logits"])
            s_list.append(o["s"])
            if return_attn:
                A_list.append(o["A"])
                B_list.append(o["B"])
        logits = torch.stack(logits_list, 0)   # (B,C)
        s = torch.stack(s_list, 0)             # (B,N)
        ret = {"logits": logits, "s": s}
        if return_attn:
            ret["A_list"] = A_list
            ret["B_list"] = B_list
        return ret

    # regularizers
    def entropy_loss(self, A_list: List[torch.Tensor]) -> torch.Tensor:
        ent = 0.0
        for A in A_list:
            ent += -(A * (A.clamp_min(1e-12).log())).sum(dim=1).mean()
        return ent / len(A_list)

    def pull_loss(self, A_list: List[torch.Tensor], Z_list: List[torch.Tensor]) -> torch.Tensor:
        P = F.normalize(self.prototypes, dim=1)         # (N,d)
        Pn2 = (P**2).sum(dim=1, keepdim=True)           # (N,1)
        loss = 0.0
        for A, Z in zip(A_list, Z_list):
            Zp = F.normalize(self.proj(Z), dim=1)       # (M,d)
            Zn2 = (Zp**2).sum(dim=1, keepdim=True)      # (M,1)
            d2 = Zn2 + Pn2.t() - 2.0*(Zp @ P.t())
            loss += (A * d2).mean()
        return loss / len(A_list)

# 2) Train / Eval that consume your loader(batch=dict)
def train_one_epoch_from_loader(
    model: PPCAC,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    in_dim: int = 1024,
    ce_weight: float = 1.0,
    ent_weight: float = 0.02,
    pull_weight: float = 0.001,
    log_interval: int = 50,
):
    model.train()
    total, correct, total_loss = 0, 0, 0.0

    for step, batch in enumerate(loader):
        feats_list, labels, metas = _to_feats_list_from_batch(batch, in_dim=in_dim)
        feats_list = [f.to(device) for f in feats_list]
        labels = labels.to(device)

        out = model(feats_list, return_attn=True)
        logits = out["logits"]
        ce = F.cross_entropy(logits, labels)

        A_list = out["A_list"]
        ent = model.entropy_loss(A_list) if ent_weight else torch.tensor(0., device=device)
        pull = model.pull_loss(A_list, feats_list) if pull_weight else torch.tensor(0., device=device)

        loss = ce_weight * ce + ent_weight * ent + pull_weight * pull

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            bs = labels.size(0)
            total += bs
            total_loss += loss.item() * bs

        if (step + 1) % log_interval == 0:
            print(f"[train] step {step+1:04d} "
                  f"loss={loss.item():.4f} (ce={ce.item():.4f}, ent={ent.item():.4f}, pull={pull.item():.4f}) "
                  f"acc={correct/max(total,1):.4f}")

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}

@torch.no_grad()
def evaluate_from_loader(
    model: PPCAC,
    loader,
    device: torch.device,
    in_dim: int = 1024,
    return_attn: bool = False,
):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    attn_dump = []

    for batch in loader:
        feats_list, labels, metas = _to_feats_list_from_batch(batch, in_dim=in_dim)
        feats_list = [f.to(device) for f in feats_list]
        labels = labels.to(device)

        out = model(feats_list, return_attn=return_attn)
        logits = out["logits"]
        ce = F.cross_entropy(logits, labels)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += ce.item() * labels.size(0)

        if return_attn:
            for A, B, meta in zip(out["A_list"], out["B_list"], metas):
                attn_dump.append({"A": A.cpu(), "B": B.cpu(), **meta})

    metrics = {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}
    if return_attn:
        metrics["attn"] = attn_dump
    return metrics

# =========================================================
# 3) Prototype init from YOUR loader
# =========================================================
@torch.no_grad()
def build_prototypes_from_loader(
    loader,
    model: PPCAC,
    in_dim: int = 1024,
    max_patches: int = 50000,
    kmeans_iters: int = 30,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Randomly sample up to `max_patches` patch features from your existing loader,
    project+normalize them, and run k-means to get (N,d) prototypes.
    """
    rng = random.Random(seed)
    samples = []
    collected = 0
    need = max_patches

    if device is None:
        device = next(model.parameters()).device

    for batch in loader:
        feats_list, _, _ = _to_feats_list_from_batch(batch, in_dim=in_dim)
        for f in feats_list:
            M = f.size(0)
            take = min(M, max(1, need // max(1, len(loader))))
            idx = torch.tensor(rng.sample(range(M), take))
            samples.append(f[idx])
            collected += take
            if collected >= max_patches:
                break
        if collected >= max_patches:
            break

    X = torch.cat(samples, 0).to(device)           # (S, in_dim)
    Z = F.normalize(model.proj(X), dim=1)          # (S, d)

    # torch k-means (simple)
    N = model.N
    # init
    inds = torch.randperm(Z.size(0), device=Z.device)[:N]
    centers = Z[inds].clone()
    for it in range(kmeans_iters):
        sim = Z @ centers.t()           # (S,N)
        assign = sim.argmax(1)          # (S,)
        new_centers = torch.zeros_like(centers)
        for k in range(N):
            m = (assign == k)
            if m.any():
                new_centers[k] = Z[m].mean(0)
            else:
                new_centers[k] = Z[torch.randint(0, Z.size(0), (1,), device=Z.device)]
        new_centers = F.normalize(new_centers, dim=1)
        if (new_centers - centers).abs().mean() < 1e-5:
            centers = new_centers
            break
        centers = new_centers
    return centers.detach().cpu()

# 4) Minimal run example (wire to your loaders)
def run(train_loader, val_loader):
    in_dim = 1024
    proj_dim = 1024
    N_proto = 16
    n_classes = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPCAC(
        in_dim=in_dim, proj_dim=proj_dim,
        num_prototypes=N_proto, num_classes=n_classes,
        tau_init=0.07, pool="mean",
        proto_init=None, proto_learnable=False,
        classifier_mode="linear",   # or "class_query"
    ).to(device)

    # --- prototype init from your loader ---
    #proto = build_prototypes_from_loader(train_loader, model, in_dim=in_dim, max_patches=30000, kmeans_iters=25, device=device)
    proto = pd.read_pickle(proto_path)['prototypes'][0,::]
    proto = torch.tensor(proto)

    with torch.no_grad():
        model.prototypes.copy_(F.normalize(proto.to(device), dim=1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, 6):
        tr = train_one_epoch_from_loader(model, train_loader, optimizer, device, in_dim=in_dim,
                                         ce_weight=1.0, ent_weight=0.02, pull_weight=0.001, log_interval=20)
        va = evaluate_from_loader(model, val_loader, device, in_dim=in_dim, return_attn=False)
        print(f"[epoch {epoch}] train={tr}  val={va}")

    # ---- attention dump (per-WSI) ----
    res = evaluate_from_loader(model, val_loader, device, in_dim=in_dim, return_attn=True)
    print(f"val acc={res['acc']:.3f}, #attn={len(res['attn'])}")
    # res["attn"][i]["A"]: (Mi, N)  patch→prototype
    # res["attn"][i]["B"]: (C,  N)  class→prototype
    return model, res

run(train_loader, val_loader)
