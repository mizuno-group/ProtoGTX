import torch
from enum import Enum
import numpy as np
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.nn as nn
import torch.optim as optim
import collections
from itertools import islice
import math

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train utils 
class TrainMode(Enum):
    WARM = 'warm'
    JOINT = 'joint'
    PUSH = 'push'
    LAST_ONLY = 'last_only'
    INIT = 'init_proto'

## Define the optimizer
def get_optim_pmil(ppnet, config):
    
    joint_optimizer_specs = [
        {
            'params': ppnet.metric_net.parameters(),
            'lr': config.joint_optimizer_lrs['metric_net'],
            'weight_decay': 1e-3
        },
        {
            'params': ppnet.prototype_vectors,
            'lr': config.joint_optimizer_lrs['prototype_vectors']
        },
        {
            'params': ppnet.last_layer.parameters(),
            'lr': config.joint_optimizer_lrs['last_layer']
        },
        {
            'params': ppnet.attention_V.parameters(),
            'lr': config.joint_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_U.parameters(),
            'lr': config.joint_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_weights.parameters(),
            'lr': config.joint_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        }
    ]
    

    warm_optimizer_specs = [
        {
            'params': ppnet.metric_net.parameters(),
            'lr': config.warm_optimizer_lrs['metric_net'],
            'weight_decay': 1e-3
        },
        {
            'params': ppnet.prototype_vectors,
            'lr': config.warm_optimizer_lrs['prototype_vectors']
        },
        {
            'params': ppnet.last_layer.parameters(),
            'lr': config.warm_optimizer_lrs['last_layer']
        },
        {
            'params': ppnet.attention_V.parameters(),
            'lr': config.warm_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_U.parameters(),
            'lr': config.warm_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_weights.parameters(),
            'lr': config.warm_optimizer_lrs['attention'],
            #'weight_decay': 1e-3
        }
    ]
    
    last_layer_optimizer_specs = [
        {
            'params': ppnet.last_layer.parameters(),
            'lr': config.last_layer_optimizer_lr['last_layer']
        },
        {
            'params': ppnet.attention_V.parameters(),
            'lr': config.last_layer_optimizer_lr['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_U.parameters(),
            'lr': config.last_layer_optimizer_lr['attention'],
            #'weight_decay': 1e-3
        },
        {
            'params': ppnet.attention_weights.parameters(),
            'lr': config.last_layer_optimizer_lr['attention'],
            #'weight_decay': 1e-3
        }
    ]

    ## define optimizer and scheduler
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=config.joint_lr_step_size,
                                                        gamma=config.joint_lr_gamma)
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    warm_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(warm_optimizer, gamma=config.warm_lr_gamma)
    
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    return joint_optimizer, joint_lr_scheduler, warm_optimizer, warm_lr_scheduler, last_layer_optimizer

def get_optim(model, args):  # for clam
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

# def _freeze_layer(layer):
#     for p in layer.parameters():
#         p.requires_grad = False

# def _unfreeze_layer(layer):
#     for p in layer.parameters():
#         p.requires_grad = True

# def last_only(model):
#     _freeze_layer(model.metric_net)
#     model.prototype_vectors.requires_grad = False

#     _unfreeze_layer(model.attention_V)
#     _unfreeze_layer(model.attention_U)
#     _unfreeze_layer(model.attention_weights)
#     _unfreeze_layer(model.last_layer)

# def joint(model):
#     _unfreeze_layer(model.metric_net)
#     model.prototype_vectors.requires_grad = True
#     _freeze_layer(model.attention_V)
#     _freeze_layer(model.attention_U)
#     _freeze_layer(model.attention_weights)
#     _freeze_layer(model.last_layer)

# def warm_only(model):
#     _unfreeze_layer(model.metric_net)
#     model.prototype_vectors.requires_grad = True
#     _unfreeze_layer(model.attention_V)
#     _unfreeze_layer(model.attention_U)
#     _unfreeze_layer(model.attention_weights)
#     _unfreeze_layer(model.last_layer)

# other utils
def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(ids) if len(ids) != 0 else 0 for ids in dataset.slide_cls_ids]
    # weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

# metrix utils
def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids

def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
   
def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)
 
 
def collate_MIL(batch):
    #img = torch.cat([item[0] for item in batch], dim = 0)
    # img = [item[0] for item in batch][0]
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    coords = [item[2] for item in batch][0]
    inst_label = [item[3] for item in batch][0]
    slide_id = [item[4] for item in batch]
    return [img, label, coords, inst_label, slide_id]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 