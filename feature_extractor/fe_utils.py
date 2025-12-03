#!/usr/bin/env python3
"""
Created on 2025-11-06 (Thu) 13:46:31

feature extraction utils

@author: I.Azuma
"""
# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import shape

# %%
# JSONを読み込む
def load_geojson(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for feature in data["features"]:
        props = feature["properties"]
        geom = feature["geometry"]

        label = props.get("classification", {}).get("name", None)
        color = props.get("classification", {}).get("color", None)
        geom_type = geom["type"]
        coords = geom["coordinates"]

        # shapelyで扱いやすい形に変換
        polygon = shape(geom)
        minx, miny, maxx, maxy = polygon.bounds
        area = polygon.area

        records.append({
            "id": feature.get("id"),
            "label": label,
            "color_rgb": color,
            "geometry_type": geom_type,
            "n_points": len(coords[0]) if geom_type == "Polygon" else None,
            "bbox_xmin": minx,
            "bbox_ymin": miny,
            "bbox_xmax": maxx,
            "bbox_ymax": maxy,
            "area": area,
            "coordinates": coords[0] if geom_type == "Polygon" else coords,
        })
    df = pd.DataFrame(records)
    return df

# bbotx: (xmin, ymin, xmax, ymax)
def intersects(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def overlapping_patches(wsi, coords, bbox, patch_size=1024, display=False):

    result = []
    result_idx = []
    for idx, (x, y) in enumerate(coords):
        patch_bbox = (x, y, x + patch_size, y + patch_size)
        if intersects(patch_bbox, bbox):
            result.append([x, y])
            result_idx.append(idx)
    coords_overlap = np.array(result)
    if display:
        for target_coords in coords_overlap:
            print(f"Overlapping patch at: {target_coords}")

            img2 = wsi.read_region((int(target_coords[0]), int(target_coords[1])), 0, (int(1024), int(1024))).convert('RGB')
            plt.imshow(img2)
            plt.axis('off')
            plt.show()
    
    return coords_overlap, result_idx

def contained_patches(wsi, coords, bbox, patch_size=1024, threshold=0.8, display=False):
    """
    coords: list[(x,y)]
    bbox: (xmin, ymin, xmax, ymax)
    threshold: 0.8 → パッチの8割以上が含まれる場合のみ採用
    """

    xmin_b, ymin_b, xmax_b, ymax_b = bbox

    selected_coords = []
    selected_indices = []

    patch_area = patch_size * patch_size

    for idx, (x, y) in enumerate(coords):

        # パッチの bbox
        ax1, ay1 = x, y
        ax2, ay2 = x + patch_size, y + patch_size

        # 交差領域の計算
        ix1 = max(ax1, xmin_b)
        iy1 = max(ay1, ymin_b)
        ix2 = min(ax2, xmax_b)
        iy2 = min(ay2, ymax_b)

        # 交差なし
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        overlap_area = (ix2 - ix1) * (iy2 - iy1)
        ratio = overlap_area / patch_area

        if ratio >= threshold:
            selected_coords.append([x, y])
            selected_indices.append(idx)
    selected_coords = np.array(selected_coords)
    if display:
        for target_coords in selected_coords:
            print(f"Overlapping patch at: {target_coords}")

            img2 = wsi.read_region((int(target_coords[0]), int(target_coords[1])), 0, (int(1024), int(1024))).convert('RGB')
            plt.imshow(img2)
            plt.axis('off')
            plt.show()

    return selected_coords, selected_indices
