import os
import re
import numpy as np

from math import sqrt
from copy import deepcopy
from logging import Logger
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import DinoVisionTransformer as Dinov2VisionTransformer
from einops import rearrange

dinov2_common_kwargs = dict(
    img_size=518,
    patch_size=14,
    mlp_ratio=4,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_fn=partial(Block, attn_class=MemEffAttention)
)

dino_common_kwargs = dict(
    num_classes=0,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_small_kwargs = dict(embed_dim=384, num_heads=6)
vit_base_kwargs = dict(embed_dim=768, num_heads=12)

MODEL_DICT = {
    "dinov2_vits14": partial(Dinov2VisionTransformer, **vit_small_kwargs, **dinov2_common_kwargs),
    "dinov2_vitb14": partial(Dinov2VisionTransformer, **vit_base_kwargs, **dinov2_common_kwargs)
}

URL_DICT = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
}

DIM_DICT = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768
}


class DINOv2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14"):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", name[:-1])  # type: nn.Module
        self.dim = DIM_DICT[name]

    def learnable_parameters(self):
        return self.dino.parameters()

    def set_requires_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, key: str = "x_norm_patchtokens", cls_key: str = "x_norm_clstoken",
                reshape: bool = False) -> tuple[torch.Tensor, ...]:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        cls_token = feature_dict[cls_key]

        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)

        return feature, cls_token


class DINOv2BackboneExpanded(nn.Module):
    def __init__(self, name: str ="dinov2_vitb14", n_splits: int = 0, mode: str = "block_expansion",
                 freeze_norm_layer: bool = True):
        super().__init__()
        self.dim = DIM_DICT[name]
        assert mode in ["block_expansion", "append"]
        expand_state_dict = block_expansion_dino if mode == "block_expansion" else append_blocks
        if n_splits > 0:
            arch = MODEL_DICT[name]
            state_dict = torch.hub.load_state_dict_from_url(URL_DICT[name], map_location="cpu")
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = expand_state_dict(
                state_dict=state_dict,
                n_splits=n_splits,
                freeze_layer_norm=freeze_norm_layer
            )
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = torch.hub.load('facebookresearch/dinov2', name[:-1])  # type: nn.Module
            self.learnable_param_names = []

    def learnable_parameters(self):
        return list(param for name, param in self.dino.named_parameters() if name in self.learnable_param_names)

    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        When appending a single block to the backbone and freezing the rest during fine-tuning, the output of the second last block
        is the original DINO feature, which will be used for DINO-specific foreground extraction during the second stage of training (for loss calculation etc.).
        An alternate approach is to cache foreground masks and part assignment maps generated from the first stage of training.
        """
        x = self.dino.prepare_tokens_with_masks(x)

        original_feature = None
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i == len(self.dino.blocks) - 2:
                original_feature = self.dino.norm(x)
        x = self.dino.norm(x)
        return x[:, self.dino.num_register_tokens + 1:, :], original_feature[:, self.dino.num_register_tokens + 1:, :], x[:, 0, :]


class DINOBackboneExpanded(nn.Module):
    def __init__(self, name: str = "dino_vitb16", n_splits: int = 0, mode: str = "block_expansion",
                 freeze_norm_layer: bool = True):
        super().__init__()
        self.dim = DIM_DICT[name]
        assert mode in ["block_expansion", "append"]
        expand_state_dict = block_expansion_dino if mode == "block_expansion" else append_blocks

        arch = MODEL_DICT[name]
        state_dict = torch.hub.load_state_dict_from_url(URL_DICT[name], map_location="cpu")
        if n_splits > 0:
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = expand_state_dict(
                state_dict=state_dict,
                n_splits=n_splits,
                freeze_layer_norm=freeze_norm_layer
            )
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = arch(depth=12)
            self.dino.load_state_dict(state_dict)
            self.learnable_param_names = []

    def learnable_parameters(self):
        return list(param for name, param in self.dino.named_parameters() if name in self.learnable_param_names)

    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def forward_with_original_feature(self, x, return_attn: bool = False):
        x = self.dino.prepare_tokens(x)

        original_feature = None
        attn_maps = []
        for i, blk in enumerate(self.dino.blocks):
            if return_attn:
                x, a = blk(x, return_attn=True)
                attn_maps.append(a)
            else:
                x = blk(x)
            if i == 11:
                original_feature = self.dino.norm(x)
        x = self.dino.norm(x)
        if return_attn:
            return x[:, 1:], original_feature[:, 1:], x[:, 0, :]
        return x[:, 1:], original_feature[:, 1:], x[:, 0, :]

    def forward(self, x):
        x = self.dino.prepare_tokens(x)
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
        x = self.dino.norm(x)
        return x[:, 1:], x[:, 0]

# %%
def block_expansion_dino(state_dict: dict[str, torch.Tensor], n_splits: int = 3, freeze_layer_norm: bool = True):
    """Perform Block Expansion on a ViT described in https://arxiv.org/abs/2404.17245"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)

    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))

    expanded_state_dict = dict()
    learnable_param_names, zero_param_names = [], []

    for dst_idx, src_idx in enumerate(block_indices.flatten()):
        src_keys = [k for k in state_dict if f"blocks.{src_idx}" in k]
        dst_keys = [k.replace(f"blocks.{src_idx}", f"blocks.{dst_idx}") for k in src_keys]

        block_state_dict = dict()

        for src_k, dst_k in zip(src_keys, dst_keys):
            if ("mlp.fc2" in dst_k or "attn.proj" in dst_k) and (dst_idx in new_block_indices):
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]

        expanded_state_dict.update(block_state_dict)

        if dst_idx in new_block_indices:
            learnable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, len(block_indices.flatten()), learnable_param_names, zero_param_names


def append_blocks(state_dict: dict[str, torch.Tensor], n_splits: int = 1, freeze_layer_norm: bool = True):
    """Append new ViT blocks with zero-ed MLPs and Attention Projection. Other weights initialized using last layer"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    src_block_idx = n_blocks - 1
    src_keys = [k for k in state_dict if f"blocks.{src_block_idx}" in k]  # keys of parameters to copy from

    expanded_state_dict = deepcopy(state_dict)
    learnable_param_names, zero_param_names = [], []
    for i in range(n_splits):
        dst_block_idx = n_blocks + i
        dst_keys = [k.replace(f"blocks.{src_block_idx}", f"blocks.{dst_block_idx}") for k in src_keys]

        block_state_dict = dict()
        for src_k, dst_k in zip(src_keys, dst_keys):
            if "mlp.fc2" in dst_k or "attn.proj" in dst_k:
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]
        expanded_state_dict.update(block_state_dict)
        learnable_param_names += dst_keys

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, n_blocks + n_splits, learnable_param_names, zero_param_names


def print_parameters(net: nn.Module, logger: Logger):
    logger.info("Learnable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            msg = name + ("(zero-ed)" if param.detach().sum() == 0 else "")
            logger.info(msg)

"""
The following functions are adapted from https://github.com/tfzhou/ProtoSeg
"""

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()  # shape: [K, B,]
    K, B = L.shape

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indices = torch.argmax(L, dim=1)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices


