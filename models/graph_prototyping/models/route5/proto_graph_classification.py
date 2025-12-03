#!/usr/bin/env python3
"""
Created on 2025-10-12 (Sun) 21:29:21

@author: I.Azuma
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class ProtoGraphClassification:
    def __init__(self):
        """Initialize the prototype graph classification system."""
        # Directory paths
        self.BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'
        self.SAVE_DIR = f'{self.BASE_DIR}/workspace3/new_model_dev/250818_graph_prototyping/route3/prototype_graphs/results'
        self.proto_path = f'{self.BASE_DIR}/github/PANTHER/src/splits/classification/CPTAC/prototypes/prototypes_c16_features_kmeans_num_1.0e+04.pkl'
        
        # Load prototype features (defined in previous analysis)
        self.proto_feats = self._load_prototype_features()
        
        # Load data
        self.train_adj_matrix_list, self.train_labels_list = self._load_train_data()
        self.val_adj_matrix_list, self.val_labels_list = self._load_val_data()
        
        # Create datasets and data loaders
        self._setup_datasets()
        
        # Initialize model
        self._setup_model()
        
        # Train the model
        self._train_model()
    
    def _load_prototype_features(self):
        """Load prototype features from pickle file."""
        loader = open(self.proto_path, 'rb')
        file = pickle.load(loader)
        loader.close()
        return file['prototypes'].squeeze()

    def _load_adj_labels(self, df):
        """Load adjacency matrices and labels from dataframe."""
        slide_list = df['slide_id'].tolist()
        slide_list = sorted([t.split('.')[0] for t in slide_list])

        class_dict = {'Normal': 0, 'LUAD': 1, 'LSCC': 2}
        adj_matrix_list = []
        labels_list = []
        for slide_id in tqdm(slide_list):
            try:
                res = pickle.load(open(f'{self.SAVE_DIR}/{slide_id}_proto_graph.pkl','rb'))
            except:
                print(f"Skip: {slide_id}")
                continue
            cooccurrence_matrix = res['cooccurrence_matrix']
            unique_labels = res['unique_labels']

            # Correct to 16x16 co-occurrence matrix (non-existing elements are 0)
            full_matrix = np.zeros((16,16))
            for i, lbl_i in enumerate(unique_labels):
                for j, lbl_j in enumerate(unique_labels):
                    full_matrix[lbl_i, lbl_j] = cooccurrence_matrix[i, j]
            adj_matrix_list.append(full_matrix)
            labels_list.append(class_dict[df[df['slide_id'].str.contains(slide_id)]['label'].values[0]])
        
        return adj_matrix_list, labels_list

    def _load_train_data(self):
        """Load training data."""
        train_df = pd.read_csv(f'{self.BASE_DIR}/workspace3/new_model_dev/250730_nonparam_prototyping/split_info/train.csv')
        return self._load_adj_labels(train_df)

    def _load_val_data(self):
        """Load validation data."""
        val_df = pd.read_csv(f'{self.BASE_DIR}/workspace3/new_model_dev/250730_nonparam_prototyping/split_info/val.csv')
        return self._load_adj_labels(val_df)

    def _setup_datasets(self):
        """Setup datasets and data loaders."""
        # Create datasets
        # proto_feats is loaded in advance
        self.pyg_dataset_train = CPTACGraphDataset(self.train_adj_matrix_list, self.train_labels_list, self.proto_feats)
        self.pyg_dataset_val = CPTACGraphDataset(self.val_adj_matrix_list, self.val_labels_list, self.proto_feats)

        # Create data loaders for PyG
        sampler = TorchRandomSampler(self.pyg_dataset_train)
        self.train_loader = PyGDataLoader(
            self.pyg_dataset_train,
            batch_size=32,
            sampler=sampler,  # Instead of shuffle=True
            collate_fn=collate_tensor_batch
        )

        val_sampler = TorchRandomSampler(self.pyg_dataset_val)
        self.val_dataloader = PyGDataLoader(
            self.pyg_dataset_val,
            batch_size=32,
            sampler=val_sampler,  # Instead of shuffle=True
            collate_fn=collate_tensor_batch
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')


class CPTACGraphDataset(Dataset):
    def __init__(self, adj_matrices, labels, proto_features):
        super().__init__()
        self.adj_matrices = adj_matrices
        self.labels = labels
        self.proto_features = torch.as_tensor(proto_features, dtype=torch.float)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        adj_matrix = torch.as_tensor(self.adj_matrices[idx], dtype=torch.float)
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        
        data = Data(x=self.proto_features, 
                    edge_index=edge_index, 
                    edge_attr=edge_weight, 
                    y=label)
        return data


class TorchRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


def collate_tensor_batch(batch):
    adjs, labels = zip(*batch)
    try:
        adj_batch = torch.stack(adjs, dim=0)
    except RuntimeError:
        max_n = max(a.size(0) for a in adjs)
        adj_batch = torch.zeros(len(adjs), max_n, max_n, dtype=torch.float)
        for i, a in enumerate(adjs):
            n = a.size(0)
            adj_batch[i, :n, :n] = a
    labels_batch = torch.stack(labels, dim=0)
    return adj_batch, labels_batch


    def _setup_model(self):
        """Setup the model."""
        # Model instantiation
        NODE_FEATURE_DIM = self.proto_feats.shape[1]
        HIDDEN_DIM = 128
        NUM_CLASSES = 3

        self.model = GraphClassifier(
            node_feature_dim=NODE_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES
        )
        print(self.model)
        self.model = self.model.to(self.device)


class GraphClassifier(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # GCN layers (considering edge weights)
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        
        # Readout layer (aggregate to graph-level vector)
        x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        
        return out

    def _train_model(self):
        """Train the model."""
        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

        n_epochs = 1000
        for epoch in range(n_epochs):
            self.model.train()
            
            train_loss = 0.0
            train_correct = 0
            
            # Extract data in batches from the data loader
            for data in self.train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.num_graphs

                pred = out.argmax(dim=1)
                train_correct += (pred == data.y).sum().item()

            # Calculate results for each epoch
            train_size = len(self.train_loader.dataset)
            train_epoch_loss = train_loss / train_size
            train_epoch_acc = train_correct / train_size

            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for data in self.val_dataloader:
                    data = data.to(self.device)
                    out = self.model(data)
                    loss = criterion(out, data.y)
                    val_loss += loss.item() * data.num_graphs
                    pred = out.argmax(dim=1)
                    val_correct += (pred == data.y).sum().item()

            # Output validation results
            val_size = len(self.val_dataloader.dataset)
            val_epoch_loss = val_loss / val_size
            val_epoch_acc = val_correct / val_size

            # Output results every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:03d} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
            
            scheduler.step()


# Example usage
if __name__ == "__main__":
    proto_classifier = ProtoGraphClassification()
