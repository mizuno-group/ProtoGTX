import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.model_sqmil import SQMIL_NIC
from models.model_protoattention import PAMIL


import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, cur):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Init Model')    
    
    # define the model for PAMIL
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
    else:
        raise NotImplementedError
    
    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.to(device)
    model.eval()
    return model

def eval_(dataset, args, ckpt_path, cur):
    print(ckpt_path)
    model = initiate_model(args, ckpt_path, cur)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _, df_inst = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df, df_inst

def summary(model, loader, args):
    n_classes = args.n_classes
    model_type = args.model_type
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_slide_ids = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader)))
    all_preds = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_id[0]
        index_label = torch.nonzero(label.squeeze(0)).to(device)
        logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=args.inst_pred, label=label)
        Y_prob = Y_prob.squeeze(0)
        # inst_label = inst_label[0]
        score = instance_dict['A_raw'].T
        
        if inst_label!=[] and sum(inst_label)!=0:
            all_slide_ids += [os.path.join(slide_id, f"{cor}") for cor in cors]  # specific for lung and renal
            inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
            all_inst_label += inst_label
            inst_score = score.detach().cpu().numpy()
            inst_score = inst_score[:, Y_hat]
            inst_score = inst_score.squeeze()
            inst_score = list((inst_score-inst_score.min())/max((inst_score.max()-inst_score.min()), 1e-10))
            inst_pred = [1 if i>0.5 else 0 for i in inst_score]
            
            all_inst_score += inst_score
            all_inst_pred += inst_pred
        
        probs = Y_prob.detach().cpu().numpy()
        acc_logger.log(Y_hat, label)
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                        'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    del data
    test_error /= len(loader)
    
    # calculate inst_auc and inst_acc
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    print("inst level aucroc: %f" % inst_auc)
    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    
    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    aucs = []
    
    if n_classes == 2:
        binary_labels = label_binarize(all_labels, classes=[0,1,2])[:,:n_classes]
    else:
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        
    for class_idx in range(n_classes):
        if class_idx in all_labels:
            fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
            aucs.append(calc_auc(fpr, tpr))
        else:
            aucs.append(float('nan'))

    auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)

    df_inst = pd.DataFrame([all_slide_ids, all_inst_label, all_inst_score,all_inst_pred]).T
    df_inst.columns = ['filename', 'label', 'prob', 'pred']
    
    return patient_results, test_error, [auc_score, inst_auc, inst_acc], df, acc_logger, df_inst
