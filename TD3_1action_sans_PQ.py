#from TD3_torch_TFE_1couche import Agent
#from TD3_torch_TFE_2couches import Agent
from TD3_torch_TFE_3couches import Agent
from TD3_torch_TFE_2couches import EarlyStopping,set_seed

import numpy as np
import pandas as pd

import grid2op
from grid2op.Reward import BaseReward
from lightsim2grid import LightSimBackend
from grid2op.Action import BaseAction
from grid2op.VoltageControler import VCFromFileAgentOverrides
from grid2op.Chronics import ChronicsHandler
from grid2op.Parameters import Parameters

from utils import plotLearning 
import torch as T
import time 

def penalty_function(x, threshold=0.05):
    z = abs(x-1) / threshold
    exponent=np.zeros(len(z))
    # Calcul de l'exposant 
    exponent = np.where(z <= 1.0, 
                        0.5 * z,      # Pente douce à l'intérieur du seuil
                        3 * z+5)     # Pente forte à l'extérieur
    
    penalty = np.exp(np.clip(exponent, None, 700))
    penalty = penalty / np.exp(3+5)

    return penalty

class VoltageReward(BaseReward):

    def __init__(self):
        super().__init__()
        self.idx_or = None
        self.idx_ex = None
        self.v_nom_list = None
        self.reward_min = -100.0

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        if is_done:
            return 0.00
        if is_illegal or has_error:
            return -1.00

        if self.idx_or is None:
            obs_ref = env.get_obs()
            idx_or, idx_ex, v_nom = get_voltage_index_observation(obs_ref, env)
            self.idx_or = np.array(idx_or, dtype=int)
            self.idx_ex = np.array(idx_ex, dtype=int)
            self.v_nom_list = np.array(v_nom)

        obs = env._last_obs
        v_origin = obs.v_or[self.idx_or]
        v_extremity = obs.v_ex[self.idx_ex]
        v_total = np.concatenate((v_origin, v_extremity))
        v_norm = v_total / self.v_nom_list

        rho = obs.rho
        penalthy_rho = 0
        if np.any(rho>1.0) :
             penalthy_rho = -0.05

        if reward_function == 'V_MSE':    
            reward = -1000.0 * np.mean((v_norm - 1.0) ** 2) 

        if reward_function == 'V_complexe':
            reward = -1*(np.sum(np.maximum(0,(rho-0.5))**2))/len(obs.v_or) + penalthy_rho -0.1* np.sum(penalty_function(v_norm,0.025))/obs.n_sub   # mettre rho en plus avec le fait qu'on mette rien si c'est en dessous de 0.8 pas de raison
          
            
        return float(reward)

def get_voltage_index_observation(obs,env) :

    list = [None] * (obs.n_sub)
    list_or = []    
    list_ex = []
    v_nom_or = []
    v_nom_ex = []

    v_nom_all = np.ones(14)*100

    for idx, value in enumerate(obs.line_or_to_subid):
        if 0 <= value <= 13 and list[value] is None:
            list[value] = idx
            list_or.append(idx)
            v_nom_or.append(v_nom_all[value])
    for idx, value in enumerate(obs.line_ex_to_subid):
            if 0 <= value <= 13 and list[value] is None:
                list[value] = idx
                list_ex.append(idx)
                v_nom_ex.append(v_nom_all[value])

    v_nom_list= np.concatenate((v_nom_or,v_nom_ex))
    return list_or,list_ex,v_nom_list

def get_state(obs,index_list_or,index_list_ex, v_nom_list) :
    P_ref=200

    idx_or = np.array(index_list_or, dtype=int)
    idx_ex = np.array(index_list_ex, dtype=int)        
    v_origin = obs.v_or[idx_or]      
    v_extremity = obs.v_ex[idx_ex]
    
    v_total = np.concatenate((v_origin, v_extremity))
    v_total = v_total / v_nom_list  
    state = np.concatenate((v_total, obs.rho))

    return state

def get_action_from_NN_output(act_NN,env,generators_without_duplicate) :

    list = generators_without_duplicate

    v_nom_all =   np.ones(14)*100
    act = ((act_NN * 0.025)+ 1) * v_nom_all[list] 

    grid2op_action = env.action_space({
        "injection": {"prod_v": act}})
    
    return grid2op_action

def poids(reward_function,obs_val,state):
    reward1=0
    reward2=0
    reward3=0

    if reward_function == 'V_MSE':    
        reward1 = -1000.0 * np.mean((state[:obs_val.n_sub] - 1.0) ** 2)

    if reward_function == 'V_complexe':
        rho = obs_val.rho
        if np.any(rho>1.0) :
             reward3 = -0.05
        reward2 = -np.sum(np.maximum(0,(rho-0.5))**2)/len(obs_val.v_or) 
        reward1 = -0.1* np.sum(penalty_function(state[:obs_val.n_sub],0.025))/obs_val.n_sub  
    return reward1,reward2,reward3

def pad_list(lst, target_len):
    return lst + [None] * (target_len - len(lst))

env_name = "l2rpn_2019" 
reward_function = 'V_complexe'


seed=1
set_seed(seed)

p = Parameters()
p.NO_OVERFLOW_DISCONNECTION = True
p.ENV_DOES_REDISPATCHING = True

env = grid2op.make(   #création de l'env d'entrainement avec les chronics d'entrainement
    env_name+"_train",
    backend=LightSimBackend(),
    action_class=BaseAction,
    chronics_handler=ChronicsHandler(),
    reward_class=VoltageReward,
    voltagecontroler_class=VCFromFileAgentOverrides,
    param=p
   )
env.seed(seed)

env_val = grid2op.make(   #création de l'env de validation avec les chronics de validation
    env_name+"_val",        
    backend=LightSimBackend(),
    action_class=BaseAction,
    chronics_handler=ChronicsHandler(),
    reward_class=VoltageReward,
    voltagecontroler_class=VCFromFileAgentOverrides,
    param=p
 )
env_val.seed(seed)

list_chronics_subpath = env.chronics_handler.subpaths
list_chronics_values = [int(subpath[-4:]) for subpath in list_chronics_subpath]


list_chronics_subpath_val = env_val.chronics_handler.subpaths
list_chronics_values_val = [int(subpath[-4:]) for subpath in list_chronics_subpath_val]

obs = env.get_obs()
index_list_or, index_list_ex,v_nom_list = get_voltage_index_observation(obs,env)

bus_lentgh = obs.n_sub # type: ignore
generators_without_duplicate = [int(x) for x in dict.fromkeys(obs.gen_to_subid)] # type: ignore
action_length = len(generators_without_duplicate) # type: ignore
input_length = bus_lentgh + len(obs.rho) # type: ignore

alpha = 0.000025
beta = 0.0001
tau = 0.001
batch_size = 128
layer1_size = 100
layer2_size = 200
layer3_size = 100
gamma = 0.0
update_actor_iter = 2
exploration_noise = 0.1
policy_noise = 0.2
noise_clip = 0.5
replay_buffer_size = 100000 
N_episodes = len(list_chronics_values)
N_avant_val = 5
episode_convergence = 0 #épisode à partir duquel l'entrainement s'est fini
learning_frequency = 5
    
patience_val = 5
min_delta_val = 10
stopper = EarlyStopping(patience=patience_val, min_delta= min_delta_val)

agent = Agent (alpha=alpha, beta=beta,gamma=gamma,update_actor_iter=update_actor_iter,max_size=replay_buffer_size, input_dims=[input_length], tau=tau,env=env,batch_size=batch_size, layer1_size=layer1_size,layer2_size=layer2_size,n_actions=action_length,exploration_noise=exploration_noise, policy_noise=policy_noise,noise_clip=noise_clip,seed=seed)

score_history = []
time_history = []
score_val_history = []
time_val_history = []



total_start_time = time.time()
for i in range(N_episodes):
    env.set_id(np.random.choice(list_chronics_values))

    obs = env.reset() 
    done = False
    score = 0.0
    step = 0
    
    if i % N_avant_val == 0 :

        agent.actor.eval()
        score_val = 0.0
        start_time_val = time.time()

        for k in range(len(list_chronics_values_val)):
            env_val.set_id(list_chronics_values_val[k])
            obs_val = env_val.reset()
            done_val = False
            while not done_val:
                state_val = get_state(obs_val,index_list_or,index_list_ex,v_nom_list)
  
                with T.no_grad():
                    act_NN_val = agent.actor(T.tensor(state_val, dtype=T.float32).to(agent.actor.device)).cpu().numpy()
                act_grid2op_val = get_action_from_NN_output(act_NN_val,env_val,generators_without_duplicate)
                obs_val, reward_val, done_val, info_val = env_val.step(act_grid2op_val)

                state_val = get_state(obs_val,index_list_or,index_list_ex,v_nom_list)
                score_val += reward_val


        stop_time_val = time.time()
        print('Validation score at episode ', i, ' : %.2f' %score_val, 'time %.2f seconds' % (stop_time_val - start_time_val))
        score_val_history.append(score_val)
        time_val_history.append(stop_time_val - start_time_val)

        if stopper.early_stop:
            print(f"Early stopping à l'épisode {i} avec un score de validation de {score_val:.2f}.")
            break

    agent.actor.train()
    
    start_time = time.time()

    while (not done) :  
        state= get_state(obs,index_list_or,index_list_ex,v_nom_list) 
  
        act_NN = agent.choose_action(state) 
        act_grid2op = get_action_from_NN_output(act_NN,env,generators_without_duplicate) 
        obs, reward, done, info = env.step(act_grid2op)
        new_state = get_state(obs,index_list_or,index_list_ex,v_nom_list)
        agent.remember (state,act_NN,reward,new_state,done)
        if step % learning_frequency == 0:
            agent.learn()

        step += 1
        score += reward 

    stop_time = time.time()       
    time_history.append(stop_time - start_time)
    episode_convergence = i + 1
    print('episode ', i, 'score %.2f' %score, 'time %.2f seconds' % (stop_time - start_time))


env._last_obs

total_stop_time = time.time()
total_time = total_stop_time - total_start_time
print('Temps total d\'entraînement : %.2f minutes' % ((total_stop_time - total_start_time)/60)) 

plotLearning(score_val_history)


