import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random
import warnings
warnings.filterwarnings("ignore")

# torch.manual_seed(hp.seed)
# torch.cuda.manual_seed(hp.seed)
# np.random.seed(hp.seed)
# random.seed(hp.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class Policy(nn.Module):
    def __init__(self, state_dim, space_dim, drop_rate, slope_rate):
        super(Policy, self).__init__()
        pass
    def forward(self, observation):
        pass
        
class Agent(nn.Module):
    def __init__(self, data_name, agent_num, state_dim, space, drop_rate, slope_rate, learn_rate, decay_rate):
        super(Agent, self).__init__()
        pass
    def forward(self, observation, epsilon):
        pass

class GAT(nn.Module):
    def __init__(self, data_name, feat_dim, layer_num, drop_rate, slope_rate, device):
        super(GAT, self).__init__()
        self.data_name = data_name
        self.drop_rate = drop_rate
        self.slope_rate = slope_rate
        self.device = device
        self.class_num = 1
        self.gat_layers = nn.Sequential()
        for i in range(layer_num):
            self.gat_layers.add_module(f'gat_layer_{i}', GATConv(feat_dim, feat_dim))
        self.gat_w = nn.Linear(feat_dim, feat_dim, bias=True)
        self.gat_vec = nn.Linear(feat_dim, 1, bias=False)
        self.transformer = nn.Linear(feat_dim, feat_dim//2, bias=True)
        self.classifier = nn.Linear(feat_dim//2, self.class_num, bias=True)
        self.loss_function = nn.BCEWithLogitsLoss()
    def build_adj(self, ins_adj, threshold):
        ins_adj[ins_adj > threshold] = 1
        ins_adj[ins_adj != 1] = 0
        ins_edges = coo_matrix(ins_adj)
        ins_edges = np.vstack((ins_edges.row, ins_edges.col))
        return ins_adj, ins_edges
    def forward(self, input):
        pass

class RGMIL(nn.Module):
    def __init__(self,
                 data_name,
                 DRL_type,
                 space_1, space_2,
                 state_1_dim, state_2_dim,
                 GNN_learn_rate, GNN_decay_rate,
                 Agent_learn_rate, Agent_decay_rate,
                 drop_rate, slope_rate,
                 discnt_rate,
                 ep_start_num, ep_end_num, ep_decay_num,
                 history_num, lambda_num,
                 device):
        super(RGMIL, self).__init__()
        self.DRL_type = DRL_type
        self.space_1 = space_1
        self.space_2 = space_2
        self.space_2_dim = len(space_2)
        self.discnt_rate = discnt_rate
        self.ep_decay_num = ep_decay_num
        self.epsilons = np.linspace(ep_start_num, ep_end_num, ep_decay_num)
        self.history_num = history_num
        self.lambda_num = lambda_num
        self.device = device
        self.current_t = None
        self.current_k = None
        self.next_k = None
        self.threshold = None
        self.layer_num = None
        self.stop_flag = False
        self.current_combination = None
        self.GNN = GAT(data_name, state_2_dim, self.space_2_dim, drop_rate, slope_rate, device).to(device)
        self.optimizer = torch.optim.Adam(self.GNN.parameters(), lr=GNN_learn_rate, weight_decay=GNN_decay_rate)
        self.agent_1 = Agent(data_name, 1, state_1_dim, space_1, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate).to(device)
        self.agent_2 = Agent(data_name, 2, state_2_dim, space_2, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate).to(device)
        self.agent_1.train()
        self.agent_2.train()
        self.history_performance = [0.]*history_num
        self.combination_record = {}
        self.reward_record = [1.]*10
        self.loss_agent1_trace = []
        self.loss_agent2_trace = []
        self.loss_total_trace = []
        self.threshold_trace = []
        self.layer_trace = []
    def copy_para(self):
        pass
    def get_target_q(self, next_observation_1, next_observation_2):
        pass
    def predict(self, observation_1, observation_2, epsilon):
        pass
    def train(self, train_loader):
        self.GNN.train()
        for bag in train_loader:
            bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
            bag_label = torch.tensor(bag_label, dtype=torch.float).unsqueeze(0).to(self.device)
            self.optimizer.zero_grad()
            bag_probs, _ = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
            loss = self.GNN.loss_function(bag_probs, bag_label)
            loss.backward()
            self.optimizer.step()
    def validate(self, val_loader):
        self.GNN.eval()
        val_acc = 0
        for bag in val_loader:
            bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
            bag_probs, _ = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
            bag_predict = 0 if F.sigmoid(bag_probs) < 0.5 else 1
            val_acc += 1.0 if bag_label == bag_predict else 0.0
        val_acc /= len(val_loader)
        history_acc = np.mean(np.array(self.history_performance[-self.history_num:]))
        self.history_performance.extend([val_acc])
        reward = val_acc - history_acc
        self.reward_record.append(reward)
        if np.abs(np.mean(np.array(self.reward_record[-self.history_num:]))) <= self.lambda_num:
            self.stop_flag = True
        return reward
    def test(self, test_loader):
        self.GNN.eval()
        test_acc = 0
        for bag in test_loader:
            bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
            bag_prob, _ = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
            bag_predict = 0 if F.sigmoid(bag_prob) < 0.5 else 1
            test_acc += 1.0 if bag_label == bag_predict else 0.0
        test_acc /= len(test_loader)
        return test_acc
    def forward(self, train_loader):
        epsilon = self.epsilons[min(self.current_t, self.ep_decay_num - 1)]
        observation_1 = torch.tensor(train_loader[self.current_k][-1][0], dtype=torch.float).to(self.device)
        observation_2 = torch.tensor(train_loader[self.current_k][-1][1], dtype=torch.float).to(self.device)
        q_values_1, q_values_2 = self.predict(observation_1, observation_2, epsilon)
        self.current_combination = str(self.threshold) + '-' + str(self.layer_num)
        if self.current_combination not in self.combination_record.keys():
            self.combination_record[self.current_combination] = 1
        else:
            self.combination_record[self.current_combination] += 1
        if self.combination_record[self.current_combination] <= self.history_num:
            self.train(train_loader[self.current_k][:-1])
        reward = self.validate(train_loader[-1][:-1])
        self.next_k = int((self.threshold + self.layer_num)) % 8 + 1
        next_observation_1 = torch.tensor(train_loader[self.next_k][-1][0], dtype=torch.float).to(self.device)
        next_observation_2 = torch.tensor(train_loader[self.next_k][-1][1], dtype=torch.float).to(self.device)
        next_q_values_1, next_q_values_2 = self.get_target_q(next_observation_1, next_observation_2)
        if self.DRL_type == 'IQL':
            pass
        elif self.DRL_type == 'VDN':
            pass
        self.threshold_trace.append(self.threshold)
        self.layer_trace.append(self.layer_num)
        self.current_k = self.next_k
        return self.stop_flag