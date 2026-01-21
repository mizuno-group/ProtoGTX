#!/usr/bin/env python3
"""
Created on 2026-01-20 (Tue) 14:38:05

TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification [NeurIPS 2021]

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/HDDX/Pathology_Graph'

import os
import sys
import csv
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report,  balanced_accuracy_score, precision_score, recall_score

import torch
import torch.nn.functional as F

sys.path.append(f'{BASE_DIR}/github/PathoGraphX/models')
from transmil.TransMIL import TransMIL
from transmil.lookahead import Lookahead
from baseline_utils.dataset_generic_npy import get_split_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, (data, label, cors, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        results_dict = model(data)
        logits = results_dict['logits']

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))

    return logits


@torch.no_grad()
def val_one_epoch(model, val_loader, device, data_type='val'):
    model.eval()
    labels = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='green')

    for i, (data, label, cors, _) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        results_dict = model(data)
        output = results_dict['logits']
        labels = torch.cat([labels, label], dim=0)
        preds = torch.cat([preds, output.detach()], dim=0)

    return preds.cpu(), labels.cpu()

def cal_metrics(logits, labels, num_classes):
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())

    # macro-average area under the cureve (AUC) scores
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())

    # weighted f1-score
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')

    # quadratic weighted Kappa
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')

    # macro specificity 
    specificity_list = []
    for class_idx in range(num_classes):
        true_positive = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() == class_idx))
        true_negative = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() != class_idx))
        false_positive = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() == class_idx))
        false_negative = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() != class_idx))

        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)

    macro_specificity = np.mean(specificity_list)

    # confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    return accuracy, auc, f1, kappa, macro_specificity, confusion_mat

def train_transmil(datasets, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = datasets
    train_loader = get_split_loader(train_dataset, training=True, testing = args.testing, weighted = False)
    val_loader = get_split_loader(val_dataset,  testing = args.testing)
    test_loader = get_split_loader(test_dataset, testing = args.testing)
    print('Done!')

    model = TransMIL(dim_in=args.embed_dim, n_classes=args.n_classes).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    base_optimizer = torch.optim.RAdam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    optimizer = Lookahead(base_optimizer, alpha=0.5, k=6)
    criterion = torch.nn.CrossEntropyLoss()

    weight_dir = os.path.join(args.save_dir, "weight")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    with open(f'{args.save_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{args.save_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

        
    max_val_accuracy = 0.0
    max_val_auc = 0.0
    max_val_f1 = 0.0
    max_val_kappa = 0.0
    max_val_specificity = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_logits = train_one_epoch(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch + 1)

        val_preds, val_labels = val_one_epoch(model=model, val_loader=val_loader, device=device, data_type='val')
        val_acc, val_auc, val_f1, val_kappa, val_specificity, val_mat = cal_metrics(val_preds, val_labels, num_classes=args.n_classes)
        print('Val\t[epoch {}] acc:{:.4f}\tauc:{:.4f}\tf1-score:{:.4f}\tkappa:{:.4f}\tspecificity:{:.4f}'.format(epoch + 1, val_acc, val_auc, val_f1, val_kappa, val_specificity))
        print('val matrix ......')
        print(val_mat)
        
        max_val_accuracy = max(max_val_accuracy, val_acc)
        max_val_auc = max(max_val_auc, val_auc)
        max_val_f1 = max(max_val_f1, val_f1)
        max_val_kappa = max(max_val_kappa, val_kappa)
        max_val_specificity = max(max_val_specificity, val_specificity)

        if max_val_accuracy == val_acc:
            print('best val acc found... save best acc weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_acc.pth")

        print('Max val accuracy: {:.4f}'.format(max_val_accuracy))
        print('Max val auc: {:.4f}'.format(max_val_auc))
        print('Max val f1: {:.4f}'.format(max_val_f1))
        print('Max val kappa: {:.4f}'.format(max_val_kappa))
        print('Max val specificity: {:.4f}'.format(max_val_specificity))
        with open(f'{args.save_dir}/val_matrix.txt', 'a') as f:
            print(epoch + 1, file=f)
            print(val_mat, file=f)

        with open(f'{args.save_dir}/results.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, val_acc, val_auc, val_f1, val_kappa, val_specificity])


def test_transmil(datasets, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, test_dataset = datasets
    test_loader = get_split_loader(test_dataset, testing = args.testing)
    print('Done!')

    model = TransMIL(dim_in=args.embed_dim, n_classes=args.n_classes).to(device)

    weight_dir = os.path.join(args.save_dir, "weight")

    checkpoint = torch.load(f"{weight_dir}/best_acc.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])
    print('Loaded weights from {}'.format(f"{weight_dir}/best_acc.pth"))

    test_preds, test_labels = val_one_epoch(model=model, val_loader=test_loader, device=device, data_type='test')
    

    # --- 各種スコア ---
    y_true = test_labels.numpy()
    y_pred = test_preds.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # --- 混同行列 ---
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # --- 表示 ---
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"F1 (macro):        {f1_macro:.4f}")
    print(f"F1 (micro):        {f1_micro:.4f}")
    print(f"F1 (weighted):     {f1_weighted:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")

    # --- 詳細レポート（各クラス別F1など） ---
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, digits=4))
