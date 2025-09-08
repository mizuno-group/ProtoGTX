## clustering twice to get the init prototype

import numpy as np
import pandas as pd
import argparse
import os
import torch
import faiss
import time
from sklearn.cluster import KMeans

from utils.utils import *
from datasets.dataset_generic_npy import get_split_loader, Generic_MIL_Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# testing = args.testing, weighted = args.weighted_sample

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
# parser.add_argument('--results_dir', default='./results', 
#                     help='results directory (default: ./results), load the checkpoint from result directory')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--model_type', type=str, choices=['wsod', 'wsod_nic','wsod_nic_single','nicwss',
                                                       'clam_sb', 'clam_mb', 'pmil', 'opmil', 'PAMIL'], 
                    default='wsod_nic', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
# parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--task', type=str)
parser.add_argument('--fea_dim', type=int, default=1024,
                     help='the original dimensions of patch embedding')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')

# special for Clustering
parser.add_argument('--num_clusters', type=int, default=8,
                    help='the number of clusters')
parser.add_argument('--num_protos', type=int, default=10, help='the number of protos')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



# creat split dir
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
os.makedirs(args.split_dir, exist_ok=True)
assert os.path.isdir(args.split_dir)

# # create result dir
# args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
# if not os.path.isdir(args.results_dir):
#     os.mkdir(args.results_dir)

def seed_torch(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

    
def reduce(args, bag_feats, k, cur):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    bag_prototypes = None
    for c in range(args.n_classes):
        prototypes = []
        semantic_shifts = []
        feats = bag_feats[c].cpu().numpy()
        
        kmeans = Kmeans(k=k, pca_dim=-1)
        kmeans.cluster(feats, seed=66)  # for reproducibility
        assignments = kmeans.labels.astype(np.int64)
        # compute the centroids for each cluster
        centroids = np.array([np.mean(feats[assignments == i], axis=0)
                            for i in range(k)])

        # compute covariance matrix for each cluster
        covariance = np.array([np.cov(feats[assignments == i].T)
                            for i in range(k)])

        prototypes.append(centroids)
        prototypes = np.array(prototypes)
        prototypes =  prototypes.reshape(-1, args.bag_fea_dim)
        print(prototypes.shape)
        prototypes = np.expand_dims(prototypes, axis=0)
        if bag_prototypes is None:
            bag_prototypes = prototypes
        else:
            bag_prototypes = np.concatenate((bag_prototypes, prototypes), axis=0)
            
    os.makedirs(f'datasets_deconf/{args.task}_{cur}', exist_ok=True)
    print(f'datasets_deconf/{args.task}_{cur}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    np.save(f'datasets_deconf/{args.task}_{cur}/train_bag_cls_agnostic_feats_proto_{k}.npy', bag_prototypes)

    del feats



def main(args):
    # define dataset
    print('Define the dataset...', end=' ')
    if args.task == 'renal_subtype_yfy':
        args.n_classes=3
        if args.model_type in ['PAMIL']:
            dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/renal_subtyping_yfy_npy.csv',
                data_dir = os.path.join(args.data_root_dir),
                data_mag = '0_1024',
                shuffle = False, 
                seed = 10, 
                print_info = True,
                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                patient_strat= False,
                ignore=[])
    elif args.task == 'renal_subtype':
        args.n_classes=3
        if args.model_type in ['PAMIL']:
            dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/renal_subtyping_npy.csv',
                data_dir = os.path.join(args.data_root_dir),
                data_mag = '1_512',
                shuffle = False, 
                seed = 10, 
                print_info = True,
                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                patient_strat= False,
                ignore=[])
                
    elif args.task == 'lung_subtype':
        args.n_classes=2
        if args.model_type in ['PAMIL']:
            dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/lung_subtyping_npy.csv',
                data_dir = os.path.join(args.data_root_dir),
                data_mag = '1_512',
                shuffle = False, 
                seed = 10, 
                print_info = True,
                label_dict = {'luad':0, 'lusc':1},
                patient_strat= False,
                ignore=[])
    else:
        raise NotImplementedError
    
    print('Done!')
    
    # define the dataloader
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    
    folds = np.arange(start, end)
    for cur in folds:
        print('\nInit Loaders...', end=' ')
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
            csv_path='{}/splits_{}.csv'.format(args.split_dir, cur))

        train_loader = get_split_loader(train_dataset, training=True, testing = args.testing, weighted = args.weighted_sample)
        
        # get the features
        print('\nClustering...', end=' ')
        cluster_centers = []
        with torch.no_grad():
            for batch_idx, (data, label, cors, inst_label, slide_id) in enumerate(train_loader):
                data_np = data.numpy()
                local_num = min(len(cors), args.num_clusters)
                kmeans = KMeans(n_clusters=local_num)
                kmeans.fit(data_np)
                batch_cluster_centers = kmeans.cluster_centers_
                cluster_centers.append(batch_cluster_centers)
            print("Done!")
            cluster_centers = np.concatenate(cluster_centers, axis=0)
            final_kmeans = KMeans(n_clusters=args.num_protos)
            final_kmeans.fit(cluster_centers)
            
            # 获取整个数据集每个聚类中心的聚类标签
            final_cluster_labels = final_kmeans.labels_

            # 打印每个聚类中心的标签
            for center_index, cluster_label in enumerate(final_cluster_labels):
                print(f"Cluster Center {center_index}: Cluster {cluster_label}")
            protos = final_kmeans.cluster_centers_
            os.makedirs(f'datasets_proto/{args.task}_{args.num_protos}_{cur}', exist_ok=True)
            np.save(f'datasets_proto/{args.task}_{args.num_protos}_{cur}/train_instance_feats_proto.npy', protos)
            # np.save(f'datasets_deconf/{args.task}_{cur}/train_bag_cls_agnostic_feats_proto_{k}.npy', protos)
            print('Done!')
        
        
if __name__ == '__main__':
    main(args)