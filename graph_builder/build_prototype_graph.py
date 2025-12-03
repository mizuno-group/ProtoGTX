#!/usr/bin/env python3
"""
Prototype Graph Builder

This module builds co-occurrence graphs based on prototype features and spatial 
neighborhood relationships in histopathology images.

Main functionalities:
1. Analyze spatial neighborhood patterns of prototype labels
2. Build directed co-occurrence graphs from label relationships
3. Visualize graphs with customizable parameters

Created on 2025-09-23 (Tue) 23:39:05
@author: I.Azuma
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict, Counter


def hex_to_rgb_mpl_255(hex_color):
    """
    Convert hex color to RGB values in 0-255 range.
    
    Args:
        hex_color (str): Hex color string (e.g., '#FF0000')
        
    Returns:
        tuple: RGB values in 0-255 range
    """
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x * 255) for x in rgb])


def get_default_cmap(n=32):
    """
    Generate default color map for prototype labels.
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        dict: Mapping from label indices to RGB colors
    """
    colors = [
        '#696969', '#556b2f', '#a0522d', '#483d8b', 
        '#008000', '#008b8b', '#000080', '#7f007f',
        '#8fbc8f', '#b03060', '#ff0000', '#ffa500',
        '#00ff00', '#8a2be2', '#00ff7f', '#FFFF54', 
        '#00ffff', '#00bfff', '#f4a460', '#adff2f',
        '#da70d6', '#b0c4de', '#ff00ff', '#1e90ff',
        '#f0e68c', '#0000ff', '#dc143c', '#90ee90',
        '#ff1493', '#7b68ee', '#ffefd5', '#ffb6c1'
    ]
    
    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict

def get_neighbor_label_frequencies(coords, labels, patch_size=512):
    """
    Calculate label frequencies of 8-connected neighboring patches.
    
    Args:
        coords (np.ndarray): Patch coordinates with shape (N, 2) - [x, y]
        labels (np.ndarray): Patch labels with shape (N,)
        patch_size (int): Size of each patch in pixels
        
    Returns:
        list: List of dictionaries containing neighbor label frequencies for each patch
    """
    # Create coordinate to index mapping
    coord_to_idx = {}
    for idx, (x, y) in enumerate(coords):
        coord_to_idx[(x, y)] = idx
    
    # 8-directional neighbor offsets
    neighbor_offsets = [
        (-patch_size, -patch_size),  # top-left
        (0, -patch_size),            # top
        (patch_size, -patch_size),   # top-right
        (-patch_size, 0),            # left
        (patch_size, 0),             # right
        (-patch_size, patch_size),   # bottom-left
        (0, patch_size),             # bottom
        (patch_size, patch_size)     # bottom-right
    ]
    
    neighbor_frequencies = []
    
    for i, (x, y) in enumerate(coords):
        neighbor_labels = []
        
        # Search 8-directional neighbors
        for dx, dy in neighbor_offsets:
            neighbor_coord = (x + dx, y + dy)
            if neighbor_coord in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_coord]
                neighbor_labels.append(labels[neighbor_idx])
        
        # Calculate label frequencies
        label_freq = Counter(neighbor_labels)
        neighbor_frequencies.append(dict(label_freq))
    
    return neighbor_frequencies

def analyze_label_neighborhood_patterns(coords, labels, neighbor_frequencies, verbose=True):
    """
    Analyze neighborhood patterns for each label type.
    
    Args:
        coords (np.ndarray): Patch coordinates
        labels (np.ndarray): Patch labels
        neighbor_frequencies (list): Neighbor label frequencies for each patch
        verbose (bool): Whether to print detailed analysis
        
    Returns:
        dict: Neighborhood pattern statistics for each label
    """
    unique_labels = np.unique(labels)
    label_patterns = defaultdict(list)
    
    # Group neighborhood frequencies by label
    for i, label in enumerate(labels):
        label_patterns[label].append(neighbor_frequencies[i])
    
    if verbose:
        print("=== Label Neighborhood Pattern Analysis ===")
    
    pattern_stats = {}
    
    for label in unique_labels:
        frequencies = label_patterns[label]
        # Aggregate all neighbor labels
        all_neighbor_labels = []
        for freq_dict in frequencies:
            for neighbor_label, count in freq_dict.items():
                all_neighbor_labels.extend([neighbor_label] * count)
        
        if all_neighbor_labels:
            neighbor_counter = Counter(all_neighbor_labels)
            total_neighbors = sum(neighbor_counter.values())
            
            # Store statistics
            pattern_stats[label] = {
                'patch_count': len(frequencies),
                'neighbor_distribution': neighbor_counter,
                'total_neighbors': total_neighbors
            }
            
            if verbose:
                print(f"\nLabel {label} ({len(frequencies)} patches):")
                for neighbor_label, count in neighbor_counter.most_common():
                    percentage = (count / total_neighbors) * 100
                    print(f"  Neighbor Label {neighbor_label}: {count} times ({percentage:.1f}%)")
    
    return pattern_stats



def build_cooccurrence_graph(labels, neighbor_frequencies, max_edges_per_node=1, min_cooccurrence=1):
    """
    Build directed co-occurrence graph from label relationships.
    
    Args:
        labels (np.ndarray): Patch labels
        neighbor_frequencies (list): Neighbor label frequencies for each patch
        max_edges_per_node (int): Maximum number of edges per node
        min_cooccurrence (int): Minimum co-occurrence count to create an edge
        
    Returns:
        tuple: (NetworkX DiGraph, co-occurrence matrix, unique labels)
    """
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Initialize co-occurrence matrix (source_label x target_label)
    cooccurrence_matrix = np.zeros((n_labels, n_labels))
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Aggregate neighborhood patterns
    for i, source_label in enumerate(labels):
        source_idx = label_to_idx[source_label]
        neighbor_freq = neighbor_frequencies[i]
        
        for target_label, count in neighbor_freq.items():
            if target_label != source_label:  # Exclude self-loops
                target_idx = label_to_idx[target_label]
                cooccurrence_matrix[source_idx, target_idx] += count
    
    # Build directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for label in unique_labels:
        G.add_node(label)
    
    # Add edges (top max_edges_per_node connections per node)
    for source_idx, source_label in enumerate(unique_labels):
        # Sort by co-occurrence count (descending)
        target_scores = []
        for target_idx, target_label in enumerate(unique_labels):
            if source_idx != target_idx:  # Exclude self-loops
                count = cooccurrence_matrix[source_idx, target_idx]
                if count >= min_cooccurrence:
                    target_scores.append((target_label, count))
        
        # Select top max_edges_per_node connections
        target_scores.sort(key=lambda x: x[1], reverse=True)
        for target_label, count in target_scores[:max_edges_per_node]:
            G.add_edge(source_label, target_label, weight=count)
    
    return G, cooccurrence_matrix, unique_labels

def build_cooccurrence_graph_from_matrix(matrix, max_edges_per_node=1):
    G = nx.Graph()
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
        # Each node adds up to max_edges_per_node edges
        neighbors = np.argsort(matrix[i])[::-1]  # sort in descending order
        edges_added = 0
        for j in neighbors:
            if i != j and matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j])
                edges_added += 1
                if edges_added >= max_edges_per_node:
                    break
    return G

def visualize_cooccurrence_graph(G, cooccurrence_matrix, unique_labels, figsize=(15, 6), 
                               node_size_multiplier=300, edge_width_multiplier=5):
    """
    Visualize co-occurrence graph with heatmap.
    
    Args:
        G (nx.DiGraph): Directed graph
        cooccurrence_matrix (np.ndarray): Co-occurrence matrix
        unique_labels (np.ndarray): Array of unique labels
        figsize (tuple): Figure size
        node_size_multiplier (int): Multiplier for node sizes
        edge_width_multiplier (int): Multiplier for edge widths
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get color map and normalize RGB values to 0-1 range
    label2color_dict = get_default_cmap(16)
    normalized_colors = {}
    for label, rgb in label2color_dict.items():
        normalized_colors[label] = tuple(c / 255.0 for c in rgb)
    
    # Graph visualization
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes with label-specific colors
    node_sizes = [len(list(G.neighbors(node))) * node_size_multiplier + 500 
                  for node in G.nodes()]
    node_colors = [normalized_colors.get(node, (0.5, 0.5, 0.5)) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=node_colors, alpha=0.7, ax=ax1)
    
    # Draw edges with weight-proportional thickness
    edges = G.edges(data=True)
    if edges:
        weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(weights)
        
        for edge in edges:
            source, target, data = edge
            weight = data['weight']
            width = (weight / max_weight) * edge_width_multiplier + 1
            nx.draw_networkx_edges(G, pos, [(source, target)], 
                                  width=width, alpha=0.6, 
                                  edge_color='darkblue',
                                  arrowsize=20, ax=ax1)
    
    # Draw labels and edge weights
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax1)
    
    if edges:
        edge_labels = {(edge[0], edge[1]): f'{edge[2]["weight"]}' for edge in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax1)

    ax1.set_title('Co-occurrence Label Relationship Graph', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Co-occurrence matrix heatmap
    im = ax2.imshow(cooccurrence_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(unique_labels)))
    ax2.set_yticks(range(len(unique_labels)))
    ax2.set_xticklabels(unique_labels, rotation=45)
    ax2.set_yticklabels(unique_labels)
    ax2.set_xlabel('Target Label')
    ax2.set_ylabel('Source Label')
    ax2.set_title('Co-occurrence Matrix (Source → Target)')

    # Display values in matrix
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if cooccurrence_matrix[i, j] > 0:
                ax2.text(j, i, f'{int(cooccurrence_matrix[i, j])}', 
                        ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=ax2, label='Co-occurrence Count')
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_graph_properties(G, verbose=True):
    """
    Analyze and display graph properties and statistics.
    
    Args:
        G (nx.DiGraph): Directed graph to analyze
        verbose (bool): Whether to print detailed analysis
        
    Returns:
        dict: Dictionary containing graph statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_out_degree': np.mean([G.out_degree(node) for node in G.nodes()]) if G.nodes() else 0,
        'avg_in_degree': np.mean([G.in_degree(node) for node in G.nodes()]) if G.nodes() else 0
    }
    
    if verbose:
        print("=== Graph Analysis Results ===")
        print(f"Number of nodes: {stats['num_nodes']}")
        print(f"Number of edges: {stats['num_edges']}")
        print(f"Average out-degree: {stats['avg_out_degree']:.2f}")
        print(f"Average in-degree: {stats['avg_in_degree']:.2f}")
    
    # Most influential nodes (high out-degree)
    out_degrees = [(node, G.out_degree(node)) for node in G.nodes()]
    out_degrees.sort(key=lambda x: x[1], reverse=True)
    stats['top_out_degree_nodes'] = out_degrees[:5]
    
    if verbose and out_degrees:
        print(f"\nTop nodes by out-degree (most connections):")
        for node, degree in out_degrees[:5]:
            print(f"  Label {node}: {degree} outgoing edges")
    
    # Most popular nodes (high in-degree)
    in_degrees = [(node, G.in_degree(node)) for node in G.nodes()]
    in_degrees.sort(key=lambda x: x[1], reverse=True)
    stats['top_in_degree_nodes'] = in_degrees[:5]
    
    if verbose and in_degrees:
        print(f"\nTop nodes by in-degree (most referenced):")
        for node, degree in in_degrees[:5]:
            print(f"  Label {node}: {degree} incoming edges")
    
    # Strongest relationships (highest edge weights)
    weighted_edges = [(edge[0], edge[1], edge[2]['weight']) for edge in G.edges(data=True)]
    weighted_edges.sort(key=lambda x: x[2], reverse=True)
    stats['strongest_edges'] = weighted_edges[:5]
    
    if verbose and weighted_edges:
        print(f"\nStrongest relationships (top 5 edges):")
        for source, target, weight in weighted_edges[:5]:
            print(f"  {source} → {target}: co-occurrence count {weight}")
    
    return stats


# Example usage and workflow functions
def build_and_visualize_prototype_graph(coords, labels, max_edges_per_node=1, 
                                       patch_size=512, figsize=(15, 6)):
    """
    Complete workflow to build and visualize prototype co-occurrence graph.
    
    Args:
        coords (np.ndarray): Patch coordinates
        labels (np.ndarray): Patch labels  
        max_edges_per_node (int): Maximum edges per node
        patch_size (int): Size of patches in pixels
        figsize (tuple): Figure size for visualization
        
    Returns:
        tuple: (graph, co-occurrence matrix, unique labels, figure)
    """
    print(f"Building prototype graph with max_edges_per_node={max_edges_per_node}")
    print(f"Total patches: {len(coords)}")
    print(f"Unique labels: {len(np.unique(labels))}")
    
    # Step 1: Calculate neighbor frequencies
    neighbor_freq_list = get_neighbor_label_frequencies(coords, labels, patch_size)
    
    # Step 2: Analyze neighborhood patterns
    pattern_stats = analyze_label_neighborhood_patterns(coords, labels, neighbor_freq_list)
    
    # Step 3: Build co-occurrence graph
    G, cooccurrence_matrix, unique_labels = build_cooccurrence_graph(
        labels, neighbor_freq_list, max_edges_per_node=max_edges_per_node
    )
    
    # Step 4: Visualize graph
    fig = visualize_cooccurrence_graph(G, cooccurrence_matrix, unique_labels, figsize)
    
    # Step 5: Analyze graph properties
    graph_stats = analyze_graph_properties(G)
    
    return G, cooccurrence_matrix, unique_labels, fig
