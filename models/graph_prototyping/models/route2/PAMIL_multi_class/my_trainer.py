#!/usr/bin/env python3
"""
Created on 2025-09-22 (Mon) 20:18:08

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/cluster/HDD/azuma/Pathology_Graph'

import os
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(f'{BASE_DIR}/github/PathoGraphX/models/graph_prototyping/models/route2/PAMIL_multi_class')
from pamil_datasets.dataset_generic import save_splits
from pamil_datasets.dataset_generic_npy import get_split_loader
from models.model_protoattention import PAMIL, push
from utils.utils import *
from utils.train_configs import *

# Import WandB logger
sys.path.append(BASE_DIR + '/github/wandb-util')
from wandbutil import WandbLogger  # type: ignore


configs = get_settings()


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    writer = None
        
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'bce':
        loss_fn = nn.functional.binary_cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    

    if args.model_type in ['PAMIL']:
        model_dict.update({'n_protos': args.num_protos})
        model_dict.update({'proto_path': os.path.join(args.proto_path, f'{args.task}_{args.num_protos}_{cur}', f'train_instance_feats_proto.npy')})
        model_dict.update({'proto_pred': args.proto_pred})
        model_dict.update({'proto_weight': args.proto_weight})
        model_dict.update({'inst_pred': args.inst_pred})
        model_dict.update({'k_sample': args.k_sample})
        model = PAMIL(**model_dict)
        
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    # WandB logger settings
    logger = WandbLogger(
        entity=configs.entity,
        project=configs.project,
        group=configs.group,
        name=configs.name,
        config=None,
    )

    if args.model_type in ['PAMIL']:
        # init the mode for training
        mode = TrainMode.WARM
        print("Warm mode for {} epochs".format(configs.num_warm_epochs))
        stop = False
        for epoch in tqdm(range(args.max_epochs)):
            if mode == TrainMode.WARM:
                warm_only(model)
                if epoch >= configs.num_warm_epochs:
                    mode = TrainMode.JOINT
                    print("Joint mode for {} epochs".format(configs.num_train_epochs - configs.num_warm_epochs))
            elif mode == TrainMode.JOINT:
                joint(model)
            
            loss_dict = train_loop_pamil(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn)

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

                stop = validate_pamil(cur, epoch, model, val_loader, args.n_classes, args, early_stopping, writer, loss_fn, args.results_dir)
            
            # WandB logging
            logger(epoch=epoch, **loss_dict)

            if epoch >= configs.num_train_epochs or stop:
                break
        
    # save the final results
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_final_checkpoint.pt".format(cur)))
    
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_final_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes, args.model_type)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, args.model_type)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)  
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
            
    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error


def train_loop_pamil(epoch, model, loader, optimizer, n_classes, args, writer = None, loss_fn=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    print('\n')
    for batch_idx, (data, label, cors, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        label_bn = torch.zeros(n_classes)
        label_bn[label.long()] = 1
        label_bn = label_bn.to(device)
        
        logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=args.inst_pred, label=label)
        Y_prob = Y_prob.squeeze(0)
        score = instance_dict['A_raw'].T

        acc_logger.log(Y_hat, label)
        
        # loss function
        loss_cls = loss_fn(Y_prob, label_bn.squeeze().float())

        # === use the inst loss like clam, v16 === 
        total_loss = loss_cls
        if args.w_er != 0:
            loss_er = instance_dict['loss_er']
            total_loss += args.w_er * loss_er
        if args.w_clst != 0:
            loss_clst = instance_dict['loss_clst']
            total_loss += args.w_clst * loss_clst
            
        if args.inst_pred:
            loss_inst = instance_dict['loss_inst']
            train_inst_loss += loss_inst.item()
            total_loss += args.w_inst * loss_inst  # v16
        elif args.attention_er:
            loss_att_er = instance_dict['loss_att_er']
            total_loss += args.w_att_er * loss_att_er  # v18
        if args.w_proto_clst != 0:
            loss_proto_clst = instance_dict['loss_proto_clst']
            total_loss += args.w_proto_clst * loss_proto_clst
        
        loss_value = total_loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_inst_loss /= len(loader)  # v15
    train_error /= len(loader)

    if epoch % 10 == 0:
        print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))

    loss_dict = {
        'train_loss': train_loss,
        'train_error': train_error,
        'train_inst_loss': train_inst_loss,
    }

    for i in range(n_classes):
        acc, correct, count= acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        loss_dict.update({f'train_class_{i}_acc': acc})

        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    
    return loss_dict
    

def validate_pamil(cur, epoch, model, loader, n_classes, args, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader),n_classes))
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, _) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            label_bn = torch.zeros(n_classes)
            label_bn[label.long()] = 1
            label_bn = label_bn.to(device)
            
            logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=False)
            Y_prob=Y_prob.squeeze(0)
            #inst_label = inst_label[0]
            inst_label = 0  # dummy
            score = instance_dict['A_raw'].T
                
            acc_logger.log(Y_hat, label)
            
            loss_cls = loss_fn(Y_prob, label_bn.squeeze().float())
            loss = loss_cls
            if args.w_er != 0:
                loss_er = instance_dict['loss_er']
                loss += args.w_er * loss_er
            if args.w_clst != 0:
                loss_clst = instance_dict['loss_clst']
                loss += args.w_clst * loss_clst
            val_loss += loss.item()

            prob[batch_idx] = Y_prob.cpu().numpy()
            
            labels[batch_idx] = label_bn.squeeze().detach().cpu().numpy()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    
    
    aucs = []
    binary_labels = labels # label_binarize(labels, classes=[i for i in range(n_classes)])
    for class_idx in range(n_classes):
        # if class_idx in labels:
        if 1 in labels[:, class_idx]:
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}')
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes, model_type):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader),n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, cors, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        label_bn = torch.zeros(n_classes)
        label_bn[label.long()] = 1
        label_bn = label_bn.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=False)
            Y_prob=Y_prob.squeeze(0)
        if model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
            score = instance_dict['A_raw'].T
            # score = F.softmax(score,dim=1)
        elif model_type in ['wsod']:
            score = logits

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label_bn.squeeze().detach().cpu().numpy()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.squeeze().detach().cpu().numpy()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    aucs = []
    binary_labels = all_labels# label_binarize(all_labels, classes=[i for i in range(n_classes)])
    # all_labels_list = 
    for class_idx in range(n_classes):
        if 1 in all_labels[:, class_idx]:
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

