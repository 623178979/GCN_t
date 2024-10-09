from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
# import gym
import time
import ddpg.corekan as core
from ddpg.utils.logx import EpochLogger
import ddpg.util as U
import os.path as osp
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.next_action_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, next_act, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_action_buf[self.ptr] = next_act
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs0=self.obs_buf[idxs],
                     obs1=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     next_actions=self.next_action_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    @property
    def nb_entries(self):
        return len(self.obs_buf)




def get_target_updates(vars, target_vars, tau):
    # logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        # logger.info('  {} <- {}'.format(target_var.name, var.name))
        target_var = var
        init_updates.append(target_var)
        target_var = (1. - tau) * target_var + tau * var
        soft_updates.append(target_var)
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return init_updates, soft_updates

class DDPG(object):
    def __init__(self, actor_critic=core.GNNActorCritic, critic = core.GNNQFunction, seed=0, 
            steps_per_epoch=4000, replay_size=int(750), gamma=0.99, tau=0.001, 
            polyak=0.995, pi_lr=1e-5, q_lr=1e-5, normalize_observations=True, batch_size=100,
            start_steps=10000, update_after=1000, update_every=50, act_noise=0.1,
            num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(), epochs=100,
            critic_l2_reg=0.,obs_dim = (11590, ),act_dim = (11590, ),feature_size=(14,)
        ):
        """
        Deep Deterministic Policy Gradient (DDPG)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, and a ``q`` module. The ``act`` method and
                ``pi`` module should accept batches of observations as inputs,
                and ``q`` should accept a batch of observations and a batch of 
                actions as inputs. When called, these should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``q``        (batch,)          | Tensor containing the current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to DDPG.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            act_noise (float): Stddev for Gaussian exploration noise added to 
                policy at training time. (At test time, no noise is added.)

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        # self.env, self.test_env = env_fn, env_fn
        # init env
        
        # self.obs_dim = self.env.observation_space.shape
        self.obs_dim = obs_dim
        # self.act_dim = self.env.action_space.shape[0]
        self.act_dim = act_dim

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = self.env.action_space.high[0]
        self.act_limit = 0.8

        # Create actor-critic module and target networks
        self.actor = actor_critic()
        self.actor_targ = deepcopy(self.actor)
        self.critic = critic()
        self.critic_targ = deepcopy(self.critic)
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.epochs = epochs
        self.tau = tau
        self.normalize_observations = normalize_observations
        self.critic_l2_reg = critic_l2_reg
        
        
        
        buffer_size = self.obs_dim + feature_size
        #ob norm
        if self.normalize_observations:
            # self.obs_rms = U.RunningMeanStd(shape=self.obs_dim)
            self.obs_rms = U.RunningMeanStd(shape=buffer_size)
        else:
            self.obs_rms = None
        # normalized_obs0 = torch.clamp(U.normalize(self.obs0, self.obs_rms))

        #return norm
        self.ret_rms = None
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.critic.q.parameters(), lr=self.q_lr, weight_decay=self.critic_l2_reg)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.actor)

        # Prepare for interaction with environment
        self.total_steps = self.steps_per_epoch * self.epochs
        self.start_time = time.time()
        # self.o, self.ep_ret, self.ep_len = self.env.reset(self.instances), 0, 0



        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False
        
        act_size = self.act_dim + (1,)
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=buffer_size, act_dim=act_size, size=self.replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.actor.pi, self.critic.q])
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

        # Set up function for computing DDPG Q-loss
    
    def setup_target_network_updates(self):
        actor_init_up, actor_soft_up = get_target_updates(self.actor.vars, self.actor_targ.vars, self.tau)
        critic_init_up, critic_soft_up = get_target_updates(self.critic.vars, self.critic_targ.vars, self.tau)
        self.target_init_up = [actor_init_up, critic_init_up]
        self.target_soft_up = [actor_soft_up, critic_soft_up]

    # def setup_popart(self):
    def compute_q(self, obs, action):
        q = self.critic.q(torch.as_tensor(obs, dtype=torch.float32).to(DEVICE), torch.as_tensor(action, dtype=torch.float32).to(DEVICE))
        return q   
    
    def compute_loss_q(self, Q0, target_Q):
        # Bellman backup for Q function
        # with torch.no_grad():
        #     self.q_pi_targ = self.critic.q(self.o2, self.actor.pi(self.o2))
            
        #     self.backup = self.r + self.gamma * target_Q

        # MSE loss against Bellman backup
        self.loss_q = ((Q0 - target_Q)**2).mean()
        # if self.critic_l2_reg > 0.:

        # Useful info for logging
        # self.loss_info = dict(QVals=self.q.detach().numpy())

        return self.loss_q


    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, obs, act, Q0):
        # Q_1 = Q0.clone()
        self.choice = torch.cat([1-self.actor.pi(obs), self.actor.pi(obs)],dim=2).to(DEVICE)
        self.choice_1 = torch.reshape(self.choice,(-1,2)).to(DEVICE)
        # self.indice = torch.cat([torch.arange(start=0,end=self.batch_size*11590).unsqueeze(-1).to(DEVICE),torch.reshape(act.type(torch.cuda.IntTensor),(-1,1))],-1).to(DEVICE)
        self.indice = torch.cat([torch.arange(start=0,end=self.batch_size*self.obs_dim[0]).unsqueeze(-1).to(DEVICE),torch.reshape(act.type(torch.IntTensor),(-1,1))],-1).to(DEVICE)
        self.decision = U.gather_nd(self.choice_1,self.indice).to(DEVICE)
        self.decision_1 = torch.reshape(self.decision,(-1,self.obs_dim[0],1)).to(DEVICE)
        self.actor_loss = -(torch.sum(torch.log(self.decision_1),1)*Q0).mean()
        # Q0_re = Q0.reshape(-1,1)

        # self.q_pi = self.critic.q(self.o, self.actor.pi(self.o))
        return self.actor_loss



    def update(self, obs, act, target_Q, Q0):
        # torch.autograd.set_detect_anomaly(True)
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(Q0=Q0 ,target_Q=target_Q)
        loss_q.backward(retain_graph=True)
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.critic.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        # with torch.autograd.set_detect_anomaly(True):
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(obs=obs,act=act,Q0=Q0)
        loss_pi.backward(retain_graph=True)
        # self.q_optimizer.step()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic.q.parameters():
            p.requires_grad = True

        # Record things
        # self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor.parameters(), self.actor_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            for q, q_targ in zip(self.critic.parameters(),self.critic_targ.parameters()):
                q_targ.data.mul_(self.polyak)
                q_targ.data.add_((1 - self.polyak) * q.data)

        return loss_pi, loss_q

    def get_action(self, o, noise_scale):
        a = self.actor.act(torch.as_tensor(o, dtype=torch.float32).to(DEVICE))
        # a += noise_scale * np.random.randn(self.act_dim)
        # return np.clip(a, -self.act_limit, self.act_limit)
        # a = a.detach().cpu().numpy()
        a = a.numpy()
        return np.clip(a, 0.2, 0.8)
        # return np.clip(a, 0.3, 0.7)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(self.instance), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            # self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


    def step(self, obs, apply_noise=True, compute_Q=True):
        # param_noise = None

        # actor = self
        if compute_Q:
            action = self.get_action(obs,0)
            q = self.compute_q(obs=obs,action=torch.as_tensor(action, dtype=torch.float32).to(DEVICE))
        else:
            action = self.get_action(obs,0)
            q = None
        action = np.clip(action, 0.2, 0.8)
        # action = np.clip(action, 0.3, 0.7)
        return action, q
    # # Prepare for interaction with environment
    # self.total_steps = self.steps_per_epoch * self.epochs
    # self.start_time = time.time()
    # self.o, self.ep_ret, self.ep_len = self.env.reset(), 0, 0

    def reset(self):
        
        pass
    
    def store_trans(self, obs0, action, reward, obs1, action_next, ins):
        # reward *= self.reward_scale
        B = obs0.shape[0]
        for b in range(B):
            self.replay_buffer.store(obs=obs0[b],act=action[b],rew=reward[b],next_obs=obs1[b],next_act=action_next[b], done=ins[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    # Main loop: collect experience in env and update/log each epoch
    def train(self,IM):
        
        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # print(type(batch['obs1']))
        IM_batch = IM[batch['done'].numpy().squeeze().astype(int)]
        
        self.obs1 = batch['obs1'].numpy()
        self.rewards = batch['rew'].numpy()
        self.next_act = batch['next_actions'].numpy()
        norm_obs1 = U.normalize(self.obs1,self.obs_rms)
        Q_obs1 = self.critic_targ(torch.as_tensor(norm_obs1, dtype=torch.float32).to(DEVICE),torch.as_tensor(self.next_act, dtype=torch.float32).to(DEVICE))
        # Q_obs1 = Q_obs1.detach().cpu().numpy()
        Q_obs1 = Q_obs1.numpy()
        target_Q = self.rewards + self.gamma * Q_obs1
        # self.obs0 = np.concatenate((batch['obs0'].numpy(), IM_batch), axis=-1)
        self.obs0 = batch['obs0'].numpy()
        norm_obs0 = U.normalize(self.obs0,self.obs_rms)
        norm_obs0 = np.concatenate((norm_obs0, IM_batch), axis=-1)
        norm_obs0 = torch.as_tensor(norm_obs0, dtype=torch.float32).to(DEVICE)

        critic_1 = deepcopy(self.critic)
        # Q0 = self.critic.q(norm_obs0,batch['act'].to(DEVICE))
        Q0 = critic_1.q(norm_obs0,batch['act'].to(DEVICE))
        print('-----------------------------------')
        print(batch['rew'].shape)
        print(batch['obs1'].shape)
        print(Q0.shape)
        print(target_Q.shape)
        print(batch['act'].shape)
        target_Q = torch.as_tensor(target_Q, dtype=torch.float32).to(DEVICE)
        actor_loss, critic_loss = self.update(obs=norm_obs0,target_Q=target_Q,Q0=Q0,act=batch['act'].to(DEVICE))
        # del Q0
        return critic_loss, actor_loss

    def save(self, save_path):
        self.logger.output_dir = save_path
        self.logger.save_state({'setcover':0},None)
    def load_pytorch_policy(self, fpath, deterministic=False):
        """ Load a pytorch policy saved with Spinning Up Logger."""
        
        fname = osp.join(fpath, 'pyt_save', 'model'+'.pt')
        print('\n\nLoading from %s.\n\n'%fname)

        self.actor = torch.load(fname)


