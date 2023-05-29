from model.RGMIL import RGMIL
import toolbox.hyper_parameter_loader as hp
import numpy as np
import torch
import random
import warnings
warnings.filterwarnings("ignore")

def run_RGMIL():
    print(hp.data_name)
    data_loader = np.load(f'./data/preprocessed/'+hp.data_name+'/0_bag_info.npz', allow_pickle=True)
    train_loader = data_loader['train']
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
                  Policy_layer_num=hp.Policy_layer_num,
                  drop_rate=hp.drop_rate,
                  slope_rate=hp.slope_rate,
                  discnt_rate=hp.discnt_rate,
                  ep_start_num=hp.ep_start_num,
                  ep_end_num=hp.ep_end_num,
                  ep_decay_num=hp.ep_decay_num,
                  history_num=hp.history_num,
                  lambda_num=hp.lambda_num,
                  memory_size=hp.memory_size,
                  memory_batch_size=hp.memory_batch_size,
                  device=hp.device)
    model.current_k = random.randint(0, hp.fold_num-2)
    for t in range(hp.timestep):
        model.current_t = t
        stop_flag = model.forward(train_loader)
        if stop_flag:
            model.loss_GNN_trace += [0.]*(len(model.threshold_trace)-len(model.loss_GNN_trace))
            break
    if hp.DRL_type == 'IQL':
        # print(model.history_performance)
        # print(model.reward_record)
        # print(model.loss_GNN_trace)
        # print(model.loss_agent1_trace)
        # print(model.loss_agent2_trace)
        # print(model.threshold_trace)
        print(model.current_combination)
        np.savez('./data/preprocessed/'+hp.data_name+f'/{hp.DRL_type}_result',
                 history_performance=model.history_performance[hp.history_num:],
                 reward_record=model.reward_record[hp.history_num:],
                 loss_GNN_trace=model.loss_GNN_trace,
                 loss_agent1_trace=model.loss_agent1_trace,
                 loss_agent2_trace=model.loss_agent2_trace,
                 threshold_trace=model.threshold_trace,
                 layer_trace=model.layer_trace,
                 best_layer_num=model.layer_num,
                 best_threshold=model.threshold)
    elif hp.DRL_type == 'VDN':
        # print(model.history_performance)
        # print(model.reward_record)
        # print(model.loss_GNN_trace)
        # print(model.loss_agent_trace)
        # print(model.threshold_trace)
        # print(model.layer_trace)
        print(model.current_combination)
        np.savez('./data/preprocessed/'+hp.data_name+f'/{hp.DRL_type}_result',
                 history_performance=model.history_performance[hp.history_num:],
                 reward_record=model.reward_record[hp.history_num:],
                 loss_GNN_trace=model.loss_GNN_trace,
                 loss_agent_trace=model.loss_agent_trace,
                 threshold_trace=model.threshold_trace,
                 layer_trace=model.layer_trace,
                 best_layer_num=model.layer_num,
                 best_threshold=model.threshold)

if __name__ == '__main__':
    if hp.data_name == 'MUSK1':
        hp.seed = 2
    elif hp.data_name == 'MUSK2':
        hp.seed = 0
    elif hp.data_name == 'FOX':
        hp.seed = 0
    elif hp.data_name == 'TIGER':
        hp.seed = 6
    elif hp.data_name == 'ELEPHANT':
        hp.seed = 2
    elif hp.data_name == 'BREAST':
        hp.seed = 2
    elif hp.data_name == 'MESSIDOR':
        hp.seed = 6
        hp.GNN_learn_rate = 0.00001
    elif hp.data_name == 'COLON':
        hp.seed = 7
        hp.Policy_layer_num = 8
    elif hp.data_name in ['NEWS6', 'NEWS14']:
        hp.seed = 8
    elif hp.data_name in ['NEWS2', 'NEWS10', 'NEWS18']:
        hp.seed = 9
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    random.seed(hp.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_RGMIL()


