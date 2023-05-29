import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DRL_type = 'VDN'
data_name = 'MUSK1'
bencmark_data = ['MUSK1', 'MUSK2', 'FOX', 'TIGER', 'ELEPHANT']
seed = None
Policy_layer_num = 7
scaling_coef = 3 # np.exp(-3)=0.04978706836786394<0.05
ins_pad_num = 2
fold_num = 10
repeat_num = 5
threshold_space = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
GNN_layer_space = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
timestep = 10000
epoch_num = 10000
patience = 20
drop_rate = 0.2
slope_rate = 0.1
lambda_num = 0.0001
ep_start_num = 1.0
ep_end_num = 0.0
ep_decay_num = 50
history_num = 10
memory_size = 20
memory_batch_size = 1
discnt_rate = 0.95
GNN_learn_rate = 0.0005
GNN_decay_rate = 0.001
Agent_learn_rate = 0.0005
Agent_decay_rate = 0.001
CNN_learn_rate = 0.0005
CNN_decay_rate = 0.001
CNN_feat_dim = 200



# NVIDIA GeForce RTX 3080
# 10240MiB
# NVIDIA-SMI 515.76
# Driver Version: 515.76
# CUDA Version: 11.7

