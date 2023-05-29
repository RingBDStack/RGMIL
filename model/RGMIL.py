import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from collections import namedtuple
import random
import warnings
warnings.filterwarnings("ignore")


Transition = namedtuple('Transition', ['states', 'actions', 'reward', 'next_states'])
class Memory(object):
    def __init__(self, memory_size, memory_batch_size):
        self.memory_size = memory_size
        self.memory_batch_size = memory_batch_size
        self.memory = []
    def save(self, states, actions, rewards, next_states):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(states, actions, rewards, next_states)
        self.memory.append(transition)
    def sample(self):
        return random.sample(self.memory, self.memory_batch_size)

class Policy(nn.Module):
    def __init__(self, state_dim, space_dim, drop_rate, slope_rate, layer_num):
        super(Policy, self).__init__()
        self.drop_rate = drop_rate
        self.slope_rate = slope_rate
        self.layer_num = layer_num
        self.linear_layers = nn.Sequential()
        for i in range(layer_num):
            self.linear_layers.add_module(f'gat_layer_{i}', nn.Linear(state_dim, state_dim, bias=True))
        self.classifier = nn.Linear(state_dim, space_dim, bias=True)
    def forward(self, observation):
        for i in range(self.layer_num):
            observation = F.leaky_relu(self.linear_layers[i](observation), negative_slope=self.slope_rate)
        q_values = F.leaky_relu(self.classifier(observation), negative_slope=self.slope_rate)
        return q_values

class Agent(nn.Module):
    def __init__(self, state_dim, space, drop_rate, slope_rate, learn_rate, decay_rate, layer_num):
        super(Agent, self).__init__()
        self.space = np.array(space)
        self.space_dim = len(space)
        self.space_prob = np.flipud(self.space / np.sum(self.space))
        self.policy = Policy(state_dim, self.space_dim, drop_rate, slope_rate, layer_num)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learn_rate, weight_decay=decay_rate)
        self.loss_mse = nn.MSELoss(reduction='mean')
    def forward(self, observation, epsilon):
        q_values = self.policy.forward(observation)
        q_values = F.softmax(q_values)
        if random.random() <= epsilon:
            # action = np.argwhere(self.space == np.random.choice(self.space))
            action = np.argwhere(self.space == np.random.choice(self.space, p=self.space_prob))
            action = list(action)[0].tolist()[0]
        else:
            action = int(torch.argmax(q_values))
        return q_values, action

class GAT(nn.Module):
    def __init__(self, data_name, feat_dim, layer_num, drop_rate, slope_rate, device):
        super(GAT, self).__init__()
        self.data_name = data_name
        self.drop_rate = drop_rate
        self.slope_rate = slope_rate
        self.device = device
        self.gat_layers = nn.Sequential()
        for i in range(layer_num//2):
            self.gat_layers.add_module(f'gat_layer_{i}', GATConv(feat_dim, feat_dim))
        self.gat_w = nn.Linear(feat_dim, feat_dim, bias=True)
        self.gat_vec = nn.Linear(feat_dim, 1, bias=False)
        self.transformer = nn.Linear(feat_dim, feat_dim//2, bias=True)
        self.classifier = nn.Linear(feat_dim//2, 1, bias=True)
        self.loss_function = nn.BCEWithLogitsLoss()
    def build_adj(self, ins_adj, threshold):
        ins_adj[ins_adj >= threshold] = 1.
        ins_adj[ins_adj != 1.] = 0.
        ins_edges = coo_matrix(ins_adj)
        ins_edges = np.vstack((ins_edges.row, ins_edges.col))
        return ins_adj, ins_edges
    def forward(self, input):
        ins_feats, ins_adj, threshold, layer_num = input
        ins_adj, ins_edges = self.build_adj(ins_adj, threshold)
        ins_feats = torch.tensor(ins_feats, dtype=torch.float).to(self.device)
        ins_feats_ = torch.tensor(ins_feats, dtype=torch.float).to(self.device)
        ins_edges = torch.tensor(ins_edges, dtype=torch.long).to(self.device)
        for i in range(layer_num):
            ins_feats = self.gat_layers[i](ins_feats, ins_edges)
            if (i+1)!=1 and (i+1)%2==0:
                ins_feats += ins_feats_
                ins_feats_ = ins_feats
            ins_feats = F.leaky_relu(ins_feats, negative_slope=self.slope_rate)
            ins_feats = F.dropout(ins_feats, self.drop_rate)
        coefficients = F.tanh(self.gat_w(ins_feats))
        coefficients = self.gat_vec(coefficients).squeeze()
        coefficients = F.softmax(coefficients)
        coefficients = coefficients.unsqueeze(-1)
        bag_feats = torch.sum(coefficients * ins_feats, dim=0)
        bag_feats_ = bag_feats.detach().cpu().numpy()
        bag_feats = F.leaky_relu(self.transformer(bag_feats), negative_slope=self.slope_rate)
        bag_probs = F.leaky_relu(self.classifier(bag_feats), negative_slope=self.slope_rate)
        return bag_probs, bag_feats_

class FCNN(nn.Module):
    def __init__(self, feat_dim, drop_rate, slope_rate, device):
        super(FCNN, self).__init__()
        self.drop_rate = drop_rate
        self.slope_rate = slope_rate
        self.device = device
        self.linear_1 = nn.Linear(feat_dim, feat_dim*2, bias=True)
        self.linear_2 = nn.Linear(feat_dim*2, feat_dim, bias=True)
        self.classifier = nn.Linear(feat_dim, 1, bias=True)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.gat_w = nn.Linear(feat_dim, feat_dim, bias=True)
        self.gat_vec = nn.Linear(feat_dim, 1, bias=False)
    def forward(self, input):
        ins_feats, _, _, _ = input
        ins_feats = torch.tensor(ins_feats, dtype=torch.float).to(self.device)
        ins_feats = F.leaky_relu(self.linear_1(ins_feats), negative_slope=self.slope_rate)
        ins_feats = F.dropout(ins_feats, self.drop_rate)
        ins_feats = F.leaky_relu(self.linear_2(ins_feats), negative_slope=self.slope_rate)
        ins_feats = F.dropout(ins_feats, self.drop_rate)
        coefficients = F.tanh(self.gat_w(ins_feats))
        coefficients = self.gat_vec(coefficients).squeeze()
        coefficients = F.softmax(coefficients)
        coefficients = coefficients.unsqueeze(-1)
        bag_feats = torch.sum(coefficients * ins_feats, dim=0)
        bag_probs = F.leaky_relu(self.classifier(bag_feats), negative_slope=self.slope_rate)
        return bag_probs, ins_feats.detach().cpu().numpy()

class RGMIL(nn.Module):
    def __init__(self,
                 data_name,
                 DRL_type,
                 space_1, space_2,
                 state_1_dim, state_2_dim,
                 GNN_learn_rate, GNN_decay_rate,
                 Agent_learn_rate, Agent_decay_rate,
                 Policy_layer_num,
                 drop_rate, slope_rate,
                 discnt_rate,
                 ep_start_num, ep_end_num, ep_decay_num,
                 history_num, lambda_num,
                 memory_size, memory_batch_size,
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
        if self.DRL_type == 'IQL':
            self.agent_1 = Agent(state_1_dim, space_1, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate, Policy_layer_num).to(device)
            self.agent_2 = Agent(state_2_dim, space_2, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate, Policy_layer_num).to(device)
        elif self.DRL_type == 'VDN':
            self.agent_1 = Agent(state_1_dim+state_2_dim, space_1, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate, Policy_layer_num).to(device)
            self.agent_2 = Agent(state_2_dim+state_1_dim, space_2, drop_rate, slope_rate, Agent_learn_rate, Agent_decay_rate, Policy_layer_num).to(device)
        self.agent_1.train()
        self.agent_2.train()
        self.copy_para()
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.history_performance = [0.]*history_num
        self.combination_record = {}
        self.reward_record = [0.]*history_num
        self.copy_t = 0
        self.memory = Memory(memory_size, memory_batch_size)
        self.memory_batch_size = memory_batch_size
        self.loss_GNN_trace = []
        self.loss_agent1_trace = [0.]*history_num
        self.loss_agent2_trace = [0.]*history_num
        self.loss_agent_trace = [0.]*history_num
        self.threshold_trace = []
        self.layer_trace = []
    def copy_para(self):
        self.agent_1_t = deepcopy(self.agent_1)
        self.agent_2_t = deepcopy(self.agent_2)
        self.agent_1_t.eval()
        self.agent_2_t.eval()
    def get_target_q(self, next_observation_1, next_observation_2):
        next_q_values_1, _ = self.agent_1_t(next_observation_1, 1e-10)
        next_q_values_2, _ = self.agent_2_t(next_observation_2, 1e-10)
        return next_q_values_1, next_q_values_2
    def predict(self, observation_1, observation_2, epsilon):
        q_values_1, action_1 = self.agent_1(observation_1, epsilon)
        q_values_2, action_2 = self.agent_2(observation_2, epsilon)
        return q_values_1, q_values_2, action_1, action_2
    def train_GNN(self, train_loader):
        self.GNN.train()
        total_loss = 0
        total_bag_num = 0
        for k in range(len(train_loader)-1):
            for bag in train_loader[k][:-1]:
                bag_label_, ins_feats, ins_adj = bag[0], bag[1], bag[2]
                bag_label = torch.tensor(bag_label_, dtype=torch.float).unsqueeze(0).to(self.device)
                self.optimizer.zero_grad()
                bag_probs, bag_feats = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
                loss = self.GNN.loss_function(bag_probs, bag_label)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach().cpu().numpy())
                total_bag_num += 1
        total_loss /= total_bag_num
        self.loss_GNN_trace.append(total_loss)
    def validate_GNN(self, val_loader):
        self.GNN.eval()
        val_acc = 0
        for bag in val_loader[-1][:-1]:
            bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
            bag_probs, _ = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
            bag_predict = 0 if F.sigmoid(bag_probs) <= 0.5 else 1
            val_acc += 1.0 if bag_label == bag_predict else 0.0
        val_acc /= len(val_loader[-1][:-1])
        history_acc = np.mean(np.array(self.history_performance[-self.history_num:]))
        self.history_performance.extend([val_acc])
        reward = val_acc - history_acc
        self.reward_record.append(reward)
        if np.abs(np.mean(np.array(self.reward_record[-self.history_num:]))) <= self.lambda_num and \
                len(set(self.threshold_trace[-self.history_num:])) == 1 and \
                len(set(self.layer_trace[-self.history_num:])) == 1:
            self.stop_flag = True
        return reward
    def test_GNN(self, test_loader):
        self.GNN.eval()
        test_acc = 0
        for bag in test_loader:
            bag_label, ins_feats, ins_adj = bag[0], bag[1], bag[2]
            bag_prob, _ = self.GNN((ins_feats, ins_adj, self.threshold, self.layer_num))
            bag_predict = 0 if F.sigmoid(bag_prob) <= 0.5 else 1
            test_acc += 1.0 if bag_label == bag_predict else 0.0
        test_acc /= len(test_loader)
        return test_acc
    def train_Agent(self):
        transitions = self.memory.sample()
        for transition in transitions:
            [observation_1, observation_2] = transition[0]
            [action_1, action_2] = transition[1]
            q_values_1, q_values_2, action_1_, action_2_ = self.predict(observation_1, observation_2, 1e-10)
            q_values_1 = F.softmax(q_values_1)
            q_values_2 = F.softmax(q_values_2)
            q_values_1 = q_values_1[action_1]
            q_values_2 = q_values_2[action_2]
            reward = transition[2]
            next_observation_1, next_observation_2 = transition[3]
            _, _, next_action_1, next_action_2 = self.predict(next_observation_1, next_observation_2, 1e-10)
            next_q_values_1, next_q_values_2 = self.get_target_q(next_observation_1, next_observation_2)
            next_q_values_1 = F.softmax(next_q_values_1)
            next_q_values_2 = F.softmax(next_q_values_2)
            next_q_values_1 = next_q_values_1[next_action_1]
            next_q_values_2 = next_q_values_2[next_action_2]
            if self.DRL_type == 'IQL':
                q_values_1_true = reward*1.1 + self.discnt_rate * next_q_values_1
                q_values_2_true = reward*0.9 + self.discnt_rate * next_q_values_2
                self.agent_1.optimizer.zero_grad()
                agent_1_loss = self.agent_1.loss_mse(q_values_1, q_values_1_true)
                agent_1_loss.backward()
                self.agent_1.optimizer.step()
                self.agent_2.optimizer.zero_grad()
                agent_2_loss = self.agent_2.loss_mse(q_values_2, q_values_2_true)
                agent_2_loss.backward()
                self.agent_2.optimizer.step()
                self.loss_agent1_trace.append(float(agent_1_loss.detach().cpu().numpy()))
                self.loss_agent2_trace.append(float(agent_2_loss.detach().cpu().numpy()))
            elif self.DRL_type == 'VDN':
                q_values_all = q_values_1 + q_values_2
                q_values_all_true = reward + self.discnt_rate*(next_q_values_1 + next_q_values_2)
                self.agent_1.optimizer.zero_grad()
                self.agent_2.optimizer.zero_grad()
                agent_all_loss = self.loss_mse(q_values_all, q_values_all_true)
                agent_all_loss.backward()
                self.agent_1.optimizer.step()
                self.agent_2.optimizer.step()
                self.loss_agent_trace.append(float(agent_all_loss.detach().cpu().numpy()))
        if self.copy_t % self.history_num == 0:
            self.copy_para()
        self.copy_t += 1
    def forward(self, train_loader):
        epsilon = self.epsilons[min(self.current_t, self.ep_decay_num - 1)]
        observation_1_ = torch.tensor(train_loader[self.current_k][-1][0], dtype=torch.float).to(self.device)
        observation_2_ = torch.tensor(train_loader[self.current_k][-1][1], dtype=torch.float).to(self.device)
        if self.DRL_type == 'IQL':
            observation_1, observation_2 = observation_1_, observation_2_
        elif self.DRL_type == 'VDN':
            observation_1 = torch.cat((observation_1_, observation_2_), 0)
            observation_2 = torch.cat((observation_2_, observation_1_), 0)
        _, _, action_1, action_2 = self.predict(observation_1, observation_2, epsilon)
        self.threshold = self.space_1[action_1]
        self.layer_num = self.space_2[action_2]
        self.current_combination = str(self.threshold) + '-' + str(self.layer_num)
        if self.current_combination not in self.combination_record.keys():
            self.combination_record[self.current_combination] = 1
        else:
            self.combination_record[self.current_combination] += 1
        if self.combination_record[self.current_combination] <= self.history_num:
            self.train_GNN(train_loader)
        reward = self.validate_GNN(train_loader)
        # self.next_k = int((self.threshold + self.layer_num)) % 8 + 1
        self.next_k = round((self.threshold + self.layer_num)) % 8 + 1
        print(self.next_k)
        next_observation_1_ = torch.tensor(train_loader[self.next_k][-1][0], dtype=torch.float).to(self.device)
        next_observation_2_ = torch.tensor(train_loader[self.next_k][-1][1], dtype=torch.float).to(self.device)
        if self.DRL_type == 'IQL':
            next_observation_1, next_observation_2 = next_observation_1_, next_observation_2_
        elif self.DRL_type == 'VDN':
            next_observation_1 = torch.cat((next_observation_1_, next_observation_2_), 0)
            next_observation_2 = torch.cat((next_observation_2_, next_observation_1_), 0)
        self.memory.save([observation_1, observation_2], [action_1, action_2], reward, [next_observation_1, next_observation_2])
        if self.current_t >= self.history_num:
            self.train_Agent()
        self.threshold_trace.append(self.threshold)
        self.layer_trace.append(self.layer_num)
        self.current_k = self.next_k
        return self.stop_flag
