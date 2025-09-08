import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import os

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_proto = 10, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(n_proto, n_classes)  # 加softmax

    def forward(self, proto, x):
        a = self.attention_a(proto)  # p * size[2]
        b = self.attention_b(x)  # n * size[2]
        proto_A = torch.mm(a, b.T)  # p * n
        proto_A = torch.transpose(proto_A, 0, 1)  # n * p
        A = self.attention_c(proto_A)  # n x n_classes
        return proto_A, A
    
# class Proto_Attn_Net_Gated(nn.Module):
#     def __init__(self, L = 10, D = 5, dropout = False, n_classes = 1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [nn.Linear(L, D),
#                             nn.Tanh()]
        
#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))

#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
        
#         self.attention_c = nn.Linear(D, n_classes)

#     def forward(self, x):
#         a = self.attention_a(x)  # n * d (n_protos // 2)
#         b = self.attention_b(x)  # n * d (n_protos // 2)
#         Attention = torch.mul(a, b)  # n * d (n_protos // 2)
#         proto_A = torch.transpose(proto_A, 0, 1)  # n * p
#         A = self.attention_c(proto_A)  # n x n_classes
#         return proto_A, A

class KL_Divergence_Loss(nn.Module):
    def __init__(self):
        super(KL_Divergence_Loss, self).__init__()

    def forward(self, logits1, logits2):
        # 计算Softmax
        probs1 = nn.Sigmoid()(logits1)
        probs2 = nn.Sigmoid()(logits2)
        
        # 计算KL散度损失
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(probs1), probs2)

        return loss

class abmil(nn.Module):
    def __init__(self, num_prototypes, num_class):
        super(abmil, self).__init__()
        self.L = num_prototypes
        self.D = num_prototypes // 2
        self.K = num_class
        self.num_prototypes = num_prototypes
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        
    def forward(self, proto_A):
        A_V = self.attention_V(proto_A)
        A_U = self.attention_U(proto_A)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        proto_score = torch.mm(A, proto_A)
        return proto_score, A, A_raw

class PAMIL(nn.Module):        
    def __init__(self, gate = True, size_arg = 'small', dropout = False, n_protos = 10, 
                 n_classes = 3, proto_path = None, proto_pred=False, proto_weight=0.5,
                 inst_pred=False, k_sample=8):
        super(PAMIL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [768, 512, 384]}
        self.abmil_size = [n_protos, n_protos // 2]
        size = self.size_dict[size_arg]
        self.n_protos = n_protos
        self.n_classes = n_classes
        self.proto_pred = proto_pred
        
        if proto_path:
            # load the proto 
            proto_init = torch.from_numpy(np.load(proto_path))
            self.proto = nn.Parameter(proto_init, requires_grad=True)
        else:
            print("Init the prototype vector")
        
        # define the model
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)
            
        self.attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, 
                                            n_proto=n_protos, n_classes = n_classes)
        
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        if proto_pred:
            self.proto_weight = proto_weight
            # v1 & v2
            proto_pred_classifier = [nn.Linear(size[1], n_classes)]
            self.proto_pred_classifier = nn.Sequential(*proto_pred_classifier)
            # # === v3 for the abmil ===
            # self.abmil = abmil(num_prototypes=n_protos, num_class=n_classes)
            # proto_bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
            # self.proto_pred_classifier = nn.ModuleList(proto_bag_classifiers)
            # # === ===
            
            self.ERloss_function = KL_Divergence_Loss()
            self.attention_er_function = nn.KLDivLoss(reduction='batchmean')
        
        # === for the inst pred, v16 ===
        self.inst_pred = inst_pred
        self.k_sample = k_sample
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        # === ===
        
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[-1] < self.k_sample:
            k = A.shape[-1]
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[-1]<self.k_sample:
            k = A.shape[-1]
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
        
    def forward(self, h, inst_pred=False, label=None):
        result_dict = {}
        device = h.device
        proto = self.fc(self.proto)
        h = self.fc(h)
        # attention
        proto_A, A_raw = self.attention_net(proto, h)  # size of proto_A is n * p
        A_raw = torch.transpose(A_raw, 1, 0)  # c * N
        A = F.softmax(A_raw, dim=1)  # softmax over N
        M = torch.mm(A, h)  # c * d
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
            
        if self.proto_pred:
            # === v1 get the proto score ===
            proto_score, _ = torch.max(proto_A, dim=0)  # 1 * p
            # # v2 use prototype to replace the patch
            # # max_values, _ = torch.max(proto_A, dim=1, keepdim=True)
            # # proto_replace = torch.where(proto_A == max_values, proto_A, torch.zeros_like(proto_A))
            # # proto_score = torch.sum(proto_replace, dim=0)
            
            proto_score = proto_score / torch.max(proto_score)
            proto_score = F.softmax(proto_score.unsqueeze(0), dim=1)  # softmax over p
            proto_M = torch.mm(proto_score, proto)  # 1 * p * p * d
            proto_logits = self.proto_pred_classifier(proto_M)
            # # === ===
            
            # # v3 use the ABMIL get the proto score
            # proto_score, p_A, p_A_raw = self.abmil(proto_A)
            # proto_score = F.softmax(proto_score, dim=1)  # softmax over p
            # proto_M = torch.mm(proto_score, proto)  # c * p * p * d
            # proto_logits = torch.empty(1, self.n_classes).float().to(device)
            # for c in range(self.n_classes):
            #     proto_logits[0, c] = self.proto_pred_classifier[c](proto_M[c])
            
            loss_er = self.ERloss_function(logits, proto_logits)
            logits = logits * (1-self.proto_weight) + proto_logits * self.proto_weight
            result_dict['loss_er'] = loss_er
            
            # # v18, must with v3
            # A_raw_cls = torch.softmax(A_raw, dim=0)  # softmax over class
            # p_A_raw_cls = torch.softmax(p_A_raw, dim=0)  # softmax over class
            # loss_att_er = self.attention_er_function(torch.log(A_raw_cls), p_A_raw_cls)
            # result_dict['loss_att_er'] = loss_att_er

        # use inst loss like clam, for exp v15
        if inst_pred:
            loss_inst = torch.tensor(0.)
            # get the topk features
            for c in range(self.n_classes):
                inst_label = label[0, c]
                instance_classifier = self.instance_classifiers[c]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, instance_classifier)
                else:
                    instance_loss, preds, targets = self.inst_eval_out(A, h, instance_classifier)
                loss_inst += instance_loss
            loss_inst /= self.n_classes
            result_dict['loss_inst'] = loss_inst
        
        loss_clst = torch.sum(torch.max(F.softmax(proto_A, dim=1), dim=1)[0]) / proto_A.shape[0]
        loss_proto_clst = torch.sum(torch.max(F.softmax(proto_A, dim=1), dim=0)[0]) / proto_A.shape[1]
        
        # logits_sigmoid = nn.Sigmoid()(logits).squeeze().detach().cpu().numpy()
        logits_softmax = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        # Y_prob = nn.Sigmoid()(logits)
        Y_prob = F.softmax(logits, dim=1)
        # Y_hat = torch.from_numpy(np.where(logits_sigmoid>0.5)[0]).cuda().unsqueeze(0)
        Y_hat = torch.argmax(logits)
        
        
        result_dict['A_raw'] = A_raw
        result_dict['proto_A'] = proto_A
        result_dict['features'] = M
        result_dict['loss_clst'] = loss_clst
        result_dict['loss_proto_clst'] = loss_proto_clst
        
        return logits, Y_prob, Y_hat, result_dict
        
# def push(loader, model, epoch, save_proto=False, result_dir=None, cur=None):  # previous push operation, 2024022
#     # get the proto_A
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     with torch.no_grad():
#         max_similarity = [0 for _ in range(model.n_protos)]
#         mid_proto = torch.zeros((model.proto.shape))
#         best_patients = ['' for _ in range(model.n_protos)]
#         best_cors = ['' for _ in range(model.n_protos)]
#         for batch_idx, (data, label, cors, inst_label, slide_id) in enumerate(loader):
#             cors = cors[0]
#             slide_id = slide_id[0]
#             inst_label = inst_label[0]
#             data, label = data.to(device), label.to(device)
#             _, _, _, instance_dict = model(data)
#             proto_A = instance_dict['proto_A']
#             # get the max similarity
#             _, indexs = torch.max(proto_A, dim=0)
#             for i in range(model.n_protos):
#                 if proto_A[indexs[i], i] > max_similarity[i]:
#                     max_similarity[i] = proto_A[indexs[i], i]
#                     mid_proto[i, :] = data[indexs[i], :]
#                     best_patients[i] = slide_id
#                     best_cors[i] = cors[indexs[i]]
                    
#         # push the patches features to the vector
#         model.proto = torch.nn.Parameter(mid_proto.to(device), requires_grad=True)
    
#         # if save the result
#         if save_proto:
#             save_push_result(epoch=epoch, patients=best_patients, 
#                              coords=best_cors, save_path=result_dir, cur=cur)
#     print('Done!')
    
# def push(loader, model, epoch, save_proto=False, result_dir=None, cur=None):
#     # get the proto_A
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # get the prototype <-> class matrix, get the prototype class
#     proto_class_matrix = model.attention_net.attention_c.weight.data
#     proto_class_matrix = proto_class_matrix.detach()
#     _, proto_class = torch.max(proto_class_matrix, axis=0)
    
#     with torch.no_grad():
#         max_similarity = [0 for _ in range(model.n_protos)]
#         mid_proto = torch.zeros((model.proto.shape))
#         best_patients = ['' for _ in range(model.n_protos)]
#         best_cors = ['' for _ in range(model.n_protos)]
#         for batch_idx, (data, label, cors, inst_label, slide_id) in enumerate(loader):
#             data, label = data.to(device), label.to(device)
#             cors = cors[0]
#             slide_id = slide_id[0]
#             label = label[0]
#             _, _, _, instance_dict = model(data)
#             proto_A = instance_dict['proto_A']
#             # get the max similarity with the same label

#             indexs = [-1 for _ in range(model.n_protos)]  # init the indexs            
#             for i in range(model.n_protos):
#                 if label == proto_class[i]:
#                     row = proto_A[:, i]
#                     max_index = torch.argmax(row)
                    
#                     while max_index in indexs:
#                         row[max_index] = -1e5
#                         max_index = torch.argmax(row)
#                     indexs[i] = max_index
                
#                     # save the max similarity patch
#                     if proto_A[indexs[i], i] > max_similarity[i]:
#                         max_similarity[i] = proto_A[indexs[i], i]
#                         mid_proto[i, :] = data[indexs[i], :]
#                         best_patients[i] = slide_id
#                         best_cors[i] = cors[indexs[i]]
#                 else:
#                     continue
                    
#         # push the patches features to the vector
#         model.proto = torch.nn.Parameter(mid_proto.to(device), requires_grad=True)
    
#         # if save the result
#         if save_proto:
#             save_push_result(epoch=epoch, patients=best_patients, coords=best_cors, save_path=result_dir, cur=cur)
#     print('Done!')

def push(loader, model, epoch, save_proto=False, result_dir=None, cur=None):  # v26
    # get the proto_A
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get the prototype <-> class matrix, get the prototype class
    proto_class_matrix = model.attention_net.attention_c.weight.data
    proto_class_matrix = proto_class_matrix.detach().cpu()
    _, proto_class = torch.max(proto_class_matrix, axis=0)
    
    with torch.no_grad():
        max_similarity = [-1e5 for _ in range(model.n_protos)]
        mid_proto = torch.zeros((model.proto.shape))
        best_patients = ['' for _ in range(model.n_protos)]
        best_cors = ['' for _ in range(model.n_protos)]
        for batch_idx, (data, label, cors, inst_label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            cors = cors[0]
            slide_id = slide_id[0]
            label = label[0]
            _, _, _, instance_dict = model(data)
            proto_A = instance_dict['proto_A']
            # get the max similarity with the same label

            indexs = [-1 for _ in range(model.n_protos)]  # init the indexs            
            for i in range(model.n_protos):
                if label == proto_class[i]:
                    # get the index 
                    p_index = np.where(inst_label == proto_class[i])[0]
                    if len(p_index) == 0:
                        continue
                    row = proto_A[:, i]
                    max_index = p_index[torch.argmax(row[p_index])]
                    
                    while max_index in indexs:
                        row[max_index] = -1e5
                        max_index = p_index[torch.argmax(row[p_index])]
                    indexs[i] = max_index
                
                    # save the max similarity patch
                    if proto_A[indexs[i], i] > max_similarity[i]:
                        max_similarity[i] = proto_A[indexs[i], i]
                        mid_proto[i, :] = data[indexs[i], :]
                        best_patients[i] = slide_id
                        best_cors[i] = cors[indexs[i]]
                else:
                    continue
                    
        # push the patches features to the vector
        model.proto = torch.nn.Parameter(mid_proto.to(device), requires_grad=True)
    
        # if save the result
        if save_proto:
            save_push_result(epoch=epoch, patients=best_patients, coords=best_cors, save_path=result_dir, cur=cur)
    print('Done!')
    
def save_push_result(epoch, patients, coords, save_path='result_folder/pmil_result/', cur=None):
    # save_path = result_folder/pmil_result/global_proto.h5
    gpf_dict = {'patients_id': patients, 'coords': coords}
    torch.save(gpf_dict, os.path.join(save_path, f"global_proto_{epoch}_s_{cur}.ckpt"))