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

#%%
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

    def __init__(self,**kwargs):
        super().__init__()
        self.idx_or = None
        self.idx_ex = None
        self.v_nom_list = None
        self.a = None 
        self.b = None
        self.reward_min = -100.0
        

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):

        if (is_illegal or has_error) and is_done :
            return -100000
        if is_done : 
            return 0

        if self.idx_or is None:
            obs_ref = env.get_obs()
            idx_or, idx_ex, v_nom = get_voltage_index_observation(obs_ref, env)
            self.idx_or = np.array(idx_or, dtype=int)
            self.idx_ex = np.array(idx_ex, dtype=int)
            self.v_nom_list = np.array(v_nom)

        obs = env._last_obs
        if self.a == None :
            self.a= np.sum(obs.gen_redispatchable)
        
        if self.b == None :
            gen_type = obs.gen_type #type: ignore
            gen_curtailable = []
            for i in range (len(gen_type)):#type: ignore
                if gen_type[i]== "wind" or gen_type[i]=="solar":#type: ignore
                    gen_curtailable.append(True)
                else : gen_curtailable.append(False)
            self.b = np.sum(gen_curtailable) 
            

        v_origin = obs.v_or[self.idx_or]
        v_extremity = obs.v_ex[self.idx_ex]
        v_total = np.concatenate((v_origin, v_extremity))
        v_norm = v_total / self.v_nom_list

        rho = obs.rho
        penalthy_rho = 0
        if np.any(rho>1.0) :
             penalthy_rho = -0.05

        sum_dispatch = 0
        abs_sum=0
        for i in range (len(obs.actual_dispatch)) :
            if obs.actual_dispatch[i]> 0:
                sum_dispatch = sum_dispatch + obs.actual_dispatch[i]
            abs_sum=abs_sum+abs(obs.actual_dispatch[i])


        p_potentielle = obs.gen_p_before_curtail.copy()
        p_réelle = obs.gen_p
        p_curtailment = p_potentielle - p_réelle 
        sum_curtailment = 0
        for i in range (len(p_curtailment)):
            if p_curtailment[i]>0:
                sum_curtailment = sum_curtailment + p_curtailment[i]

        if reward_function == 'V_redispatch_curtailment':
            reward = 1*(-100*(np.sum(np.maximum(0,(rho-0.5))**2))/len(obs.v_or) + 100*penalthy_rho -10* np.sum(penalty_function(v_norm,0.025))/obs.n_sub - 0.01* (sum_dispatch )/self.a  - 1* sum_curtailment/self.b)
        if is_illegal or has_error:
            reward = reward -100000
    
        
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

def get_state(obs,index_list_or,index_list_ex, v_nom_list,p_max,p_min,ramp_up,ramp_down) :

    P_ref = 200.0

    idx_or = np.array(index_list_or, dtype=int)
    idx_ex = np.array(index_list_ex, dtype=int)        
    v_origin = obs.v_or[idx_or]      
    v_extremity = obs.v_ex[idx_ex]
    v_total = np.concatenate((v_origin, v_extremity))
    v_total = v_total / v_nom_list  

    p_prod = obs.gen_p/P_ref
    p_load = obs.load_p/P_ref
    q_prod = obs.gen_q/P_ref
    q_load = obs.load_q/P_ref

    power_data = np.concatenate((p_prod,p_load,q_prod,q_load))

    gen_ids = [1,2,3,4]
    actual_redispatch = obs.actual_dispatch[gen_ids]/P_ref
    target_redispatch = obs.target_dispatch[gen_ids]/P_ref
    margin_up = np.maximum(0,np.minimum((p_max[gen_ids]*1)-obs.gen_p[gen_ids],ramp_up[gen_ids]))/P_ref
    margin_down = np.maximum(0,np.minimum(obs.gen_p[gen_ids]-(p_min[gen_ids]+p_max[gen_ids]*0),ramp_down[gen_ids]))/P_ref
    effective_margin_up = np.minimum(margin_up,np.maximum(0,(p_max[gen_ids]-p_min[gen_ids])/P_ref-actual_redispatch))
    effective_margin_down = np.minimum(margin_down, np.maximum(0, actual_redispatch -  (p_min[gen_ids] - p_max[gen_ids])/P_ref))
    redispatchable_data = np.concatenate((actual_redispatch,target_redispatch, effective_margin_up, effective_margin_down))

    gen_ids = [0]
    curtailment_data = obs.curtailment_limit[gen_ids]

    voltage_line_data = np.concatenate((v_total, obs.rho))
    state = np.concatenate((voltage_line_data,power_data,redispatchable_data,curtailment_data))

    return state

def get_action_from_NN_output(act_NN,env,generators_without_duplicate,p_max,p_min,ramp_up,ramp_down) :
    obs = env._last_obs 
    list = generators_without_duplicate
    v_nom_all = np.ones(14)*100
    act = ((act_NN[0:5] * 0.025)+ 1) * v_nom_all[list] 

    gen_ids_curtailment = 0
    # On ramène la sortie [-1, 1] du NN vers pour le curtailment
    curtailment_value = (act_NN[8]+1)/2

    gen_ids_redispatch = [1,2,3,4]
    p_max_redispatch = p_max[gen_ids_redispatch]
    p_min_redispatch = p_min[gen_ids_redispatch]
    ramp_up_redispatch = ramp_up[gen_ids_redispatch]
    ramp_down_redispatch = ramp_down[gen_ids_redispatch]
    current_redispatch = obs.target_dispatch

    nn_redispatch = act_NN[5:8]
    margin_up = np.maximum(0,np.minimum(p_max_redispatch - obs.gen_p[gen_ids_redispatch], ramp_up_redispatch))
    margin_down = np.maximum(0,np.minimum(obs.gen_p[gen_ids_redispatch] - p_min_redispatch, ramp_down_redispatch))
    effective_margin_up = np.minimum(margin_up,np.maximum(p_max_redispatch-p_min_redispatch - current_redispatch[gen_ids_redispatch],0))
    effective_margin_down = np.minimum(margin_down,np.maximum((current_redispatch[gen_ids_redispatch]-(p_min_redispatch - p_max_redispatch) ),0))

    redispatch_values = np.zeros(4)
    for i in range(0,3):
        if nn_redispatch[i] > 0:
            # Si le NN demande d'augmenter, on map vers [0, margin_up]
            redispatch_values[i] = nn_redispatch[i] * effective_margin_up[i]
        else:
            # Si le NN demande de diminuer, on map [-1, 0] vers [-margin_down, 0]
            redispatch_values[i] = nn_redispatch[i] * effective_margin_down[i]

    target_gen4 = -np.sum(redispatch_values[0:3])
    
    redispatch_values[3] = np.clip(target_gen4, -effective_margin_down[3], effective_margin_up[3])

    # Calcul du surplus (ce que Gen 4 n'a pas pu absorber)
    # Si deficit > 0 : Gen 4 n'a pas pu monter assez (le reste doit être baissé ailleurs)
    # Si deficit < 0 : Gen 4 n'a pas pu descendre assez (le reste doit être monté ailleurs)
    deficit = target_gen4 - redispatch_values[3]


    if abs(deficit) > 1e-9: 
        for i in range(0, 3):
                if deficit > 0:
                    # Gen 4 n'a pas pu MONTER assez.
                    # -> Les autres ont trop baissé. Il faut faire MONTER les autres.
                    potential_increase = effective_margin_up[i] - redispatch_values[i]
                    
                    amount_to_add = min(deficit, potential_increase)
                    redispatch_values[i] += amount_to_add
                    deficit -= amount_to_add
                    
                elif deficit < 0:
                    # Gen 4 n'a pas pu DESCENDRE assez.
                    # -> Les autres ont trop monté. Il faut faire BAISSER les autres.
                    potential_decrease = redispatch_values[i] - (-effective_margin_down[i])
                    
                    amount_to_shave = min(abs(deficit), potential_decrease)
                    redispatch_values[i] -= amount_to_shave
                    deficit += amount_to_shave 
                    
                if abs(deficit) < 1e-9: 
                    break


    dispatch_dict = {int(g): float(v) for g, v in zip(gen_ids_redispatch, redispatch_values)}

    grid2op_action = env.action_space({
        "injection": {"prod_v": act},
        "curtail":(gen_ids_curtailment,curtailment_value),
        "redispatch":dispatch_dict
        })

    

    return grid2op_action,redispatch_values,curtailment_value

def pad_list(lst, target_len):
    return lst + [None] * (target_len - len(lst))

def equilibre_poids (obs_val,index_or,index_ex,v_nom,is_illegal):
    reward_illegal = 0
    if is_illegal :
        reward_illegal = -100000.00 

    obs = obs_val
    v_origin = obs.v_or[index_or]
    v_extremity = obs.v_ex[index_ex]
    v_total = np.concatenate((v_origin, v_extremity))
    v_norm = v_total / v_nom

    rho = obs.rho
    reward_rho = -100 * (np.sum(np.maximum(0,(rho-0.5))**2))/len(obs.v_or)

    reward_v = -10* np.sum(penalty_function(v_norm,0.025))/obs.n_sub

    if (np.any(obs.rho>1)):
        penalty = -0.05*100
    else : penalty = 0


    sum_dispatch = 0
    error = 0
    for i in range (len(obs.actual_dispatch)) :
        if obs.actual_dispatch[i]> 0:
            sum_dispatch = sum_dispatch + obs.actual_dispatch[i]
        error = error + (obs.actual_dispatch[i])
    a= np.sum(obs.gen_redispatchable) 
    reward_dispatch = -0.01*(sum_dispatch)/a

    p_potentielle = obs.gen_p_before_curtail.copy()
    p_suppr=1
    p_réelle = obs.gen_p
    p_curtailment = p_potentielle - p_réelle 
    gen_type = obs.gen_type #type: ignore
    gen_curtailable = []
    for i in range (len(gen_type)):#type: ignore
        if gen_type[i]== "wind" or gen_type[i]=="solar":#type: ignore
            gen_curtailable.append(True)
        else : gen_curtailable.append(False)
    b = np.sum(gen_curtailable)
    sum_curtailment = 0

    for i in range (len(p_curtailment)):
        if p_curtailment[i]>0:
            sum_curtailment = sum_curtailment + p_curtailment[i]
    reward_curtailment = -1 * sum_curtailment/b

    equil = np.sum(obs_val.gen_p)-np.sum(obs_val.load_p)
    return reward_illegal,reward_rho,reward_v,reward_dispatch,reward_curtailment,sum_dispatch/a, abs(error)/a,p_suppr/a,equil/a,penalty

     

#%%

env_name = "l2rpn_2019" 
reward_function = 'V_redispatch_curtailment'

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

p_max = (env.gen_pmax)*[1,1,1,1,1] #type:ignore
p_min = env.gen_pmin
ramp_up = env.action_space.gen_max_ramp_up
print(ramp_up)
ramp_down = env.action_space.gen_max_ramp_down
print(ramp_down)
print(f"Redispatchable vector : {obs.gen_redispatchable}")#type: ignore
redispatchable = np.sum(obs.gen_redispatchable) #type: ignore
print(f"Redispatchable number : {redispatchable}")

gen_type = obs.gen_type #type: ignore
gen_curtailable = []
for i in range (len(gen_type)):#type: ignore
    if gen_type[i]== "wind" or gen_type[i]=="solar":#type: ignore
        gen_curtailable.append(True)
    else : gen_curtailable.append(False)
print(f"Curtailable vector : {gen_curtailable}")
curtailable = np.sum(gen_curtailable)
print(f"Curtailable : {curtailable}")

load_shedding = len(obs.load_p) #type: ignore



bus_lentgh = obs.n_sub # type: ignore
generators_without_duplicate = [int(x) for x in dict.fromkeys(obs.gen_to_subid)] # type: ignore
action_length = len(generators_without_duplicate) + redispatchable-1 + curtailable  # type: ignore 
print(f"Actions : {action_length}")
input_length = bus_lentgh + len(obs.rho) + len(obs.gen_p) + len(obs.gen_q) + len(obs.load_p) + len(obs.load_q) + 4*redispatchable + curtailable  # type: ignore      
print(f"Inputs : {input_length}")

alpha = 0.000025
beta = 0.0001
tau = 0.001
batch_size = 128
layer1_size = 100
layer2_size = 200
layer3_size = 100
gamma = 0.99
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
#%%
score_history = []
time_history = []
score_val_history = []
time_val_history = []

total_start_time = time.time()
#%% Agent classique
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
        exception=0
        for k in range(len(list_chronics_values_val)):
            env_val.set_id(list_chronics_values_val[k])
            obs_val = env_val.reset()
            done_val = False
            step = 0
            while not done_val:
                state_val = get_state(obs_val,index_list_or,index_list_ex,v_nom_list,p_max,p_min,ramp_up,ramp_down)

                with T.no_grad():
                    act_NN_val = agent.actor(T.tensor(state_val, dtype=T.float32).to(agent.actor.device)).cpu().numpy()
                act_grid2op_val,redispatch_values,curtailment= get_action_from_NN_output(act_NN_val,env_val,generators_without_duplicate,p_max,p_min,ramp_up,ramp_down)

                obs_val, reward_val, done_val, info_val = env_val.step(act_grid2op_val)

                score_val += reward_val
                step=step+1
            
        stop_time_val = time.time()
        print('Validation score at episode ', i, ' : %.2f' %score_val, 'time %.2f seconds' % (stop_time_val - start_time_val))
        score_val_history.append(score_val)
        time_val_history.append(stop_time_val - start_time_val)
        if stopper.early_stop or i==71:
            print(f"Early stopping à l'épisode {i} avec un score de validation de {score_val:.2f}.")
            break
  
    agent.actor.train()
    
    start_time = time.time()

    while (not done) :  
        state= get_state(obs,index_list_or,index_list_ex,v_nom_list,p_max,p_min,ramp_up,ramp_down) 
        act_NN = agent.choose_action(state) 
        act_grid2op,_ ,_= get_action_from_NN_output(act_NN,env,generators_without_duplicate,p_max,p_min,ramp_up,ramp_down) 
        obs, reward, done, info = env.step(act_grid2op)
        new_state = get_state(obs,index_list_or,index_list_ex,v_nom_list,p_max,p_min,ramp_up,ramp_down)
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
