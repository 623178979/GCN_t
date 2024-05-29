from copy import deepcopy
import ecole.scip
import numpy as np
import torch
from torch.optim import Adam
import time
import ctypes
from memory_profiler import profile
import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
import random
import csv

import pyscipopt as scip
import utilities
import ecole
from pathlib import Path

from ddpg.core import GNNActorCritic
from ddpg.ddpgclass import DDPG

scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
}
ec_env = ecole.environment.Branching(
    observation_function = ecole.observation.NodeBipartite(),
    scip_params=scip_parameters,
)
exec = ctypes.CDLL('/home/yunbo/workspace/GCN_t/execute.so')
exec.executeBranchRule.argtypes = [ctypes.py_object, ctypes.c_char_p, ctypes.c_bool]
exec.executeBranchRule.restype = ctypes.c_int
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# @profile
def make_samples(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    episode, instance, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    # print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance)) 

    if eval_flag==1:
        seed=0
    else:
        seed=0
  
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_params(m, seed=seed)
    m.setIntParam('timing/clocktype', 1)
    m.setRealParam('limits/time', time_limit) 

        
    varss = [x for x in m.getVars()]             
        
    
    if eval_flag==1:

        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k
        for i in min_k[0]:
            a,b = m.fixVar(varss[i],obs[i])        
        
    else:

#        minimum_k = np.where(np.random.binomial(1, actions) <0.5)
        minimum_k = np.where(np.array(actions.squeeze())<0.5)
        max_k = np.where(np.array(actions.squeeze())>0.5)[0]
        min_k = minimum_k[0]
        for i in min_k:
            a,b = m.fixVar(varss[i],obs[i])     
   
    m.optimize()
    
    K = [m.getVal(x) for x in m.getVars()]  
        
    obj = m.getObjVal()

    # print(m.getStatus())

    m.freeProb()    

    out_queue = {
        'type': 'solution',
        'episode': episode,
        'instance': instance,
        'sol' : np.array(K),
        'obj' : obj,
        'seed': seed,
        'mask': max_k,
    }            

    # print("[w {}] episode {} done".format(os.getpid(),episode))
    
    return out_queue

# @profile
def send_orders(instances, epi, obs, actions, seed, exploration_policy, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    orders_queue = []
    for i in range(len(instances)):
        seed = rng.randint(2**32)
        orders_queue.append([epi[i], instances[i], obs[i], actions[i], seed, exploration_policy, eval_flag, time_limit, out_dir])

    return orders_queue

# @profile
def collect_samples(instances, epi, obs, actions, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)
    
    pars = send_orders(instances, epi, obs, actions, rng.randint(2**32), exploration_policy, eval_flag, time_limit, tmp_samples_dir) 
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples(pars[n])
        out_Q.append(out_queue)        


    # record answers 
    i = 0
    collecter=[]
    epi=[]
    obje=[]
    instances=[]
    mask=[]

    for sample in out_Q:
        
        collecter.append(sample['sol'])
        
        epi.append(sample['episode'])
        
        obje.append(sample['obj'])

        instances.append(sample['instance'])

        mask.append(sample['mask'])
        
        i += 1

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
   
    return np.stack(collecter), np.stack(epi), np.stack(obje), instances, mask
    
# @profile
class SamplingAgent0(scip.Branchrule):

    def __init__(self, episode, instance, seed, exploration_policy, out_dir):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.exploration_policy = exploration_policy
        self.out_dir = out_dir

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):
        
        # custom policy branching           
        if self.model.getNNodes() == 1:    
            self.model_ptr = self.model.to_ptr(True)
            self.model_1 = ecole.scip.Model.from_file(self.instance)
            # self.model_1 = ecole.scip.Model.from_pyscipopt(self.model)
            # extract formula features
            self.state = utilities.extract_state(self.model_1, self.state_buffer)
            # self.state = ecole.observation.NodeBipartite().extract()            
            # print(self.state)
            
            self.name = ctypes.c_char_p(self.exploration_policy.encode('utf-8'))
            result_code = exec.executeBranchRule(self.model_ptr, self.name, allowaddcons)
            
            # result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
            # exec.getState.argtypes = [ctypes.py_object, ctypes.py_object]
            # exec.getState.restype = ctypes.py_object
            # nrows = exec.getState(self.model_ptr, self.state)
            
            # print('row num:',nrows)
                               
        elif self.model.getNNodes() != 1:
            self.model_ptr = self.model.to_ptr(True)
            self.name = ctypes.c_char_p(self.exploration_policy.encode('utf-8'))
            result_code = exec.executeBranchRule(self.model_ptr, self.name, allowaddcons)
            # result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)   
        else:
            raise NotImplementedError
        result = utilities.get_scip_result(result_code=result_code)
        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1
        
        return {'result': result}

# @profile
def make_samples0(in_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    episode, instance, seed, exploration_policy, eval_flag, time_limit, out_dir = in_queue
    # print('[w {}] episode {}, seed {}, processing instance \'{}\'...'.format(os.getpid(),episode,seed,instance))

    if eval_flag==1:
        seed=0
    else:
        seed=0
    
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem('{}'.format(instance))
    utilities.init_scip_paramsH(m, seed=seed)
    m.setIntParam('timing/clocktype', 2)
    m.setLongintParam('limits/nodes', 1) 
#    m.setRealParam('limits/time', 50) 

    branchrule = SamplingAgent0(
        episode=episode,
        instance=instance,
        seed=seed,
        exploration_policy=exploration_policy,
        out_dir=out_dir)

    m.includeBranchrule(
        branchrule=branchrule,
        name="Sampling branching rule", desc="",
        priority=666666, maxdepth=-1, maxbounddist=1)
    abc=time.time()    
    m.optimize()       
    # print(time.time()-abc)    

    b_obj = m.getObjVal()

    K = [m.getVal(x) for x in m.getVars()]     

    out_queue = {
        'type': 'formula',
        'episode': episode,
        'instance': instance,
        'state' : branchrule.state,
        'seed': seed,
        'b_obj': b_obj,
        'sol' : np.array(K),        
    }  
       
    m.freeTransform()  
        
    obj = [x.getObj() for x in m.getVars()]  
    
    out_queue['obj'] = sum(obj) 
    
    m.freeProb()  
        
    # print("[w {}] episode {} done".format(os.getpid(),episode))
    # m.re
    return out_queue

# @profile
def send_orders0(instances, n_samples, seed, exploration_policy, batch_id, eval_flag, time_limit, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    st = batch_id*n_samples
    orders_queue = []
    for i in instances[st:st+n_samples]:     
        seed = rng.randint(2**32)
        orders_queue.append([episode, i, seed, exploration_policy, eval_flag, time_limit, out_dir])
        episode += 1
    return orders_queue


# @profile
def collect_samples0(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, batch_id, eval_flag, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    tmp_samples_dir = '{}/tmp'.format(out_dir)
    os.makedirs(tmp_samples_dir, exist_ok=True)
    
    pars = send_orders0(instances, n_samples, rng.randint(2**32), exploration_policy, batch_id, eval_flag, time_limit, tmp_samples_dir)  
    
    out_Q = []
    for n in range(n_samples):
        out_queue = make_samples0(pars[n])
        out_Q.append(out_queue)        
        

    # record answers and write samples
    i = 0
    collecter=[]

    collecterM=[]

    epi=[]
    instances=[]
    obje=[]
    bobj=[]
    ini_sol=[]
    
    
    for sample in out_Q:
        # print(sample)
        ini_sol.append(sample['sol'])         
        
        collecter.append(sample['state'][2]['values'])
        
#        print(sample['state'][2]['values'].shape, sample['episode'])

        collecterM.append(np.transpose(sample['state'][1]['incidence']))
        
        epi.append(sample['episode'])
        
        instances.append(sample['instance'])
        
        obje.append(sample['obj'])

        bobj.append(sample['b_obj'])
        
        i += 1
    
    shap = np.stack(collecter).shape

    X=np.stack(collecter).reshape(-1,13)
    # print(X)
    print(np.var(X, axis=0))

    feats = X[:,[0,1,3,4,5,6,8,9,12]]

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
    

    del out_Q
    # print(feats.reshape(shap[0],shap[1],-1)[0][0])
    return feats.reshape(shap[0],shap[1],-1), np.stack(epi), np.stack(obje), np.stack(bobj), instances, np.stack(ini_sol), np.stack(collecterM)
    




# @profile
def learn(args,network='gnn',
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=25,
          nb_rollout_steps=30,
          reward_scale=1.0,
          noise_type=None,
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=1,#6, # per epoch cycle and MPI worker,
          nb_eval_steps=50,
          batch_size=32, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=30,
        #   save_path = './GCN_t/model',
          save_path = None,
        #   load_path = None,
          load_path = './GCN_t/model',
          env = ec_env
          ):
    
    batch_sample = 25
    batch_sample_eval = 10
    eval_val = 0
    time_limit = 2
    exploration_strategy = 'relpscost'

    instances_valid = []
    instances_train = glob.glob('/home/yunbo/workspace/GCN_t/data/instances/setcover/train_500r_1000c_0.05d/*.lp')
    # instances_train = glob.glob('/home/yunbo/workspace/Learn-LNS-policy/LNS_SC/data/instances/setcover/transfer_5000r_1000c_0.05d/*.lp')
    instances_valid += ['/home/yunbo/workspace/GCN_t/data/instances/setcover/test_500r_1000c_0.05d/instance_{}.lp'.format(i+1) for i in range(10)]
    # instances_valid += ['/home/yunbo/workspace/Learn-LNS-policy/LNS_SC/data/instances/setcover/validation5000//instance_{}.lp'.format(i+1) for i in range(10)]
    out_dir = '/home/yunbo/workspace/test'

    nb_epochs = 20 #20
    nb_epoch_cycles = 10#len(instances_train)//batch_sample
    
    
    nb_rollout_steps = 16 #30

    # print("{} train instances for {} samples".format(len(instances_train),nb_epoch_cycles*nb_epochs*batch_sample))



    actor_critic = GNNActorCritic

    agent = DDPG(actor_critic=actor_critic,batch_size=batch_size,critic_l2_reg=critic_l2_reg,pi_lr=actor_lr,q_lr=critic_lr,tau=tau,gamma=gamma,replay_size=1000)
    if load_path is not None:
        agent.load_pytorch_policy(load_path)

    rng = np.random.RandomState(args.seed)

    nenvs = batch_sample

    t = 0

    min_obj = 10000

    for epoch in range(1):
        random.shuffle(instances_train)

        fieldnames = [
            'instance',
            'obj',
            'initial',
            'bestroot',
            'imp',
            'mean',
            'time',
            'seed',
            'size',
        ]
        print('epoch loop')
        result = "test_{}.csv".format(time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs('test_results', exist_ok=True)
        with open("test_results/{}".format(result),'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for cycle in range(1):
                print('epoch',epoch,'cycle',cycle)
                # evalue
                if cycle%1==0:
                    episodes = 0
                    t = 0
                    

                    for cyc in range(len(instances_valid)//batch_sample_eval):
                        a_1 = time.time()
                        obj_list = []
                        formu_feat, epi, ori_objs, best_root, instances, ini_sol, IM=collect_samples0(instances_valid, out_dir + '/train', rng, batch_sample_eval,
                                        args.njobs, exploration_policy=exploration_strategy,
                                        batch_id=cyc,
                                        eval_flag=1,
                                        time_limit=None)
                        print('epoch',epoch,'cycle',cycle,'cyc',cyc)

                        init_sols = ini_sol

                        ori_objs = np.copy(best_root)
                        current_sols = init_sols
                        record_ini = np.copy(ori_objs)

                        pre_sols = np.zeros([2, batch_sample_eval, 1000])
                        rec_inc = [[] for r in range(batch_sample_eval)]
                        [rec_inc[r].append(init_sols[r]) for r in range(batch_sample_eval)]
                        rec_best = np.copy(best_root)
                        incu_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)])
                        incu_val_avg = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)])
                        current_obs = np.concatenate((formu_feat, incu_val[:,:,np.newaxis], incu_val_avg[:,:,np.newaxis], pre_sols.transpose(1,2,0), current_sols[:,:,np.newaxis]), axis=-1)

                        for t_rollout in range(nb_eval_steps):
                            action, q = agent.step(np.concatenate((current_obs, IM), axis=-1))

                            pre_sols = np.concatenate((pre_sols,current_sols[np.newaxis,:,:]), axis=0)


                            action = np.random.binomial(1,action)
                            action = np.where(action > 0.5, action, 0.)
                            action = np.where(action == 0., action, 1.)


                            current_sols, epi, current_objs, instances, mask = collect_samples(instances, epi, current_sols, action, out_dir + '/train', rng, batch_sample_eval,
                                    args.njobs, exploration_policy=exploration_strategy,
                                    eval_flag=1,
                                    time_limit=time_limit) 
                            # print('epoch',epoch,'cycle',cycle,'eve_t_roll',t_rollout)
                            inc_ind = np.where(current_objs < rec_best)[0]
                            [rec_inc[r].append(current_sols[r]) for r in inc_ind]
                            rec_best[inc_ind] = current_objs[inc_ind]
                            re = ori_objs - current_objs

                            t += 1

                            inc_val = np.stack([rec_inc[r][-1] for r in range(batch_sample_eval)])
                            avg_inc_val = np.stack([np.array(rec_inc[r]).mean(0) for r in range(batch_sample_eval)])         
        
                            next_obs = np.concatenate((formu_feat, inc_val[:,:,np.newaxis], avg_inc_val[:,:,np.newaxis], pre_sols[-2:].transpose(1,2,0), current_sols[:,:,np.newaxis]), axis=-1)

                            current_obs = next_obs
                            ori_objs = current_objs
                            obj_list.append(current_objs)
                            if time.time()-a_1 > 2000.0:
                                print('break')
                                break

                        # print(time.time()-a_1)
                        ti = time.time()-a_1
                        miniu = np.stack(obj_list).min(axis=0)  
                        ave = np.mean(miniu)
                        for j in range(batch_sample_eval):                 
                            writer.writerow({
                                'instance': instances[j],
                                'obj':miniu[j],
                                'initial':record_ini[j],
                                'bestroot':best_root[j],
                                'imp':best_root[j]-miniu[j],
                                'mean':ave,
                                'time':ti,
                                'seed':args.seed,
                                'size':1000
                            })
                            csvfile.flush()

                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    
    parser.add_argument(
        '-t', '--total_timesteps',
        help='Number of total_timesteps.',
        type=int,
        default=1e4,
    )
    arg = parser.parse_args()
    learn(args=arg)
