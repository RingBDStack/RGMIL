from model.RGMIL import RGMIL, GAT, FCNN
import toolbox.hyper_parameter_loader as hp
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
from random import choice
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def run_RGMIL(repeat_idx):
    data_loader = np.load(f'./data/preprocessed/' + hp.data_name + '/0_bag_info.npz', allow_pickle=True)
    train_loader = data_loader['train']
    print(hp.data_name)
    state_1_dim = train_loader[0][-1][0].shape[0]
    state_2_dim = train_loader[0][-1][1].shape[0]
    model = RGMIL(data_name=hp.data_name,
                  DRL_type=hp.DRL_type,
                  space_1=hp.threshold_space,
                  space_2=hp.GNN_layer_space,
                  state_1_dim=state_1_dim,
                  state_2_dim=state_2_dim,
                  GNN_learn_rate=hp.GNN_learn_rate,
                  GNN_decay_rate=hp.GNN_decay_rate,
                  Agent_learn_rate=hp.Agent_learn_rate,
                  Agent_decay_rate=hp.Agent_decay_rate,
                  drop_rate=hp.drop_rate,
                  slope_rate=hp.slope_rate,
                  discnt_rate=hp.discnt_rate,
                  ep_start_num=hp.ep_start_num,
                  ep_end_num=hp.ep_end_num,
                  ep_decay_num=hp.ep_decay_num,
                  history_num = hp.history_num,
                  lambda_num = hp.lambda_num,
                  device=hp.device)
    model.current_k = random.randint(0, hp.fold_num-2)
    for t in range(hp.timestep):
        model.current_t = t
        stop_flag = model.forward(train_loader)
        if stop_flag:
            break
    print(model.current_combination)
    best_layer_num = model.layer_num
    best_threshold = model.threshold

    best_layer_num = 1
    best_threshold = 1
    test_acc_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    test_auc_list = []
    for fold_idx in range(hp.fold_num):
        data_loader = np.load(f'./data/preprocessed/' + hp.data_name + f'/{fold_idx}_bag_info.npz', allow_pickle=True)
        train_loader = data_loader['train']
        test_loader = data_loader['test']
        GNN = GAT(hp.data_name, state_2_dim, best_layer_num, hp.drop_rate, hp.slope_rate, hp.device).to(hp.device)
        optimizer = torch.optim.Adam(GNN.parameters(), lr=hp.GNN_learn_rate, weight_decay=hp.GNN_decay_rate)
        best_test_acc = 0.0
        best_test_precision = 0.0
        best_test_recall = 0.0
        best_test_f1 = 0.0
        best_test_auc = 0.0
        patience = 0
        for epoch_idx in range(hp.epoch_num):
            GNN.train()
            train_loss = 0.0
            train_pred_list = []
            train_true_list = []
            for k in range(9):
                for bag in train_loader[k][:-1]:
                    bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
                    bag_label_ = torch.tensor(bag_label, dtype=torch.float).unsqueeze(0).to(hp.device)
                    optimizer.zero_grad()
                    bag_probs, _ = GNN((ins_feats, ins_adj, best_threshold, best_layer_num))
                    bag_predict = 0 if F.sigmoid(bag_probs) < 0.5 else 1
                    train_pred_list.append(bag_predict)
                    train_true_list.append(bag_label)
                    loss = GNN.loss_function(bag_probs, bag_label_)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss
            train_pred_list = np.array(train_pred_list)
            train_true_list = np.array(train_true_list)
            train_acc = accuracy_score(train_true_list, train_pred_list)
            GNN.eval()
            test_loss = 0.0
            test_pred_list = []
            test_prob_list = []
            test_true_list = []
            for bag in test_loader:
                bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
                bag_label_ = torch.tensor(bag_label, dtype=torch.float).unsqueeze(0).to(hp.device)
                bag_probs, _ = GNN((ins_feats, ins_adj, best_threshold, best_layer_num))
                bag_predict = 0 if F.sigmoid(bag_probs) < 0.5 else 1
                test_pred_list.append(bag_predict)
                test_prob_list.append(F.sigmoid(bag_probs))
                test_true_list.append(bag_label)
                loss = GNN.loss_function(bag_probs, bag_label_)
                test_loss += loss
            test_pred_list = np.array(test_pred_list)
            test_true_list = np.array(test_true_list)
            test_acc = accuracy_score(test_true_list, test_pred_list)
            if hp.data_name == 'COLON':
                test_precision = precision_score(test_true_list, test_pred_list)
                test_recall = recall_score(test_true_list, test_pred_list)
                test_f1 = f1_score(test_true_list, test_pred_list)
                test_auc = roc_auc_score(test_true_list, test_prob_list)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if hp.data_name == 'COLON':
                    best_test_precision = test_precision
                    best_test_recall = test_recall
                    best_test_f1 = test_f1
                    best_test_auc = test_auc
                patience = 0
            else:
                patience += 1
                if patience >= hp.patience:
                    test_acc_list.append(best_test_acc)
                    if hp.data_name == 'COLON':
                        test_precision_list.append(best_test_precision)
                        test_recall_list.append(best_test_recall)
                        test_f1_list.append(best_test_f1)
                        test_auc_list.append(best_test_auc)
                    break
    x1 = np.array(test_acc_list)
    print('Acc: {:.5f}'.format(np.mean(x1)) + '±{:.5f}'.format(np.std(x1)))
    if hp.data_name == 'COLON':
        x2 = np.array(test_precision_list)
        x3 = np.array(test_recall_list)
        x4 = np.array(test_f1_list)
        x5 = np.array(test_auc_list)
        print('Precision: {:.5f}'.format(np.mean(x2)) + '±{:.5f}'.format(np.std(x2)))
        print('Recall: {:.5f}'.format(np.mean(x3)) + '±{:.5f}'.format(np.std(x3)))
        print('F1: {:.5f}'.format(np.mean(x4)) + '±{:.5f}'.format(np.std(x4)))
        print('AUC: {:.5f}'.format(np.mean(x5)) + '±{:.5f}'.format(np.std(x5)))


if __name__ == '__main__':
    for repeat_idx in range(hp.repeat_num):
        run_RGMIL(repeat_idx)
        print('-----------------------------------------------')
