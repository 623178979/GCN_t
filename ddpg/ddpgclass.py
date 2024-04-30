from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import ddpg.core as core
from ddpg.utils.logx import EpochLogger


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
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

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
    def __init__(self, env_fn, actor_critic=core.GNNActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, tau=0.001, 
            polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
            update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
            max_ep_len=1000, logger_kwargs=dict(), save_freq=1, instances=None):
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

        self.env, self.test_env = env_fn, env_fn
        # init env
        
        # self.obs_dim = self.env.observation_space.shape
        self.obs_dim = (1000, )
        # self.act_dim = self.env.action_space.shape[0]
        self.act_dim = (1000, )

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # act_limit = self.env.action_space.high[0]
        self.act_limit = 0.8

        # Create actor-critic module and target networks
        self.ac = actor_critic()
        self.ac_targ = deepcopy(self.ac)
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
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
        self.instances = instances
        self.tau = tau

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Prepare for interaction with environment
        self.total_steps = self.steps_per_epoch * self.epochs
        self.start_time = time.time()
        self.o, self.ep_ret, self.ep_len = self.env.reset(self.instances), 0, 0



        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

        # Set up function for computing DDPG Q-loss
    
    def setup_target_network_updates(self):
        actor_init_up, actor_soft_up = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_up, critic_soft_up = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_up = [actor_init_up, critic_init_up]
        self.target_soft_up = [actor_soft_up, critic_soft_up]

    # def setup_popart(self):
        
    
    def compute_loss_q(self, data):
        self.o, self.a, self.r, self.o2, self.d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        self.q = self.ac.q(self.o,self.a)

        # Bellman backup for Q function
        with torch.no_grad():
            self.q_pi_targ = self.ac_targ.q(self.o2, self.ac_targ.pi(self.o2))
            self.backup = self.r + self.gamma * self.q_pi_targ

        # MSE loss against Bellman backup
        self.loss_q = ((self.q - self.backup)**2).mean()

        # Useful info for logging
        self.loss_info = dict(QVals=self.q.detach().numpy())

        return self.loss_q, self.loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        self.o = data['obs']
        self.q_pi = self.ac.q(self.o, self.ac.pi(self.o))
        return -self.q_pi.mean()

    # # Set up optimizers for policy and q-function
    # self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
    # self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

    # # Set up model saving
    # logger.setup_pytorch_saver(self.ac)

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        # return np.clip(a, -self.act_limit, self.act_limit)
        return np.clip(a, 0.2, 0.8)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(self.instance), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # def setup_popart(self):
    #     self

    # def step(self, IM):
    #     batch = self.
    # # Prepare for interaction with environment
    # self.total_steps = self.steps_per_epoch * self.epochs
    # self.start_time = time.time()
    # self.o, self.ep_ret, self.ep_len = self.env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    def train(self,IM):
        for t in range(self.total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy (with some noise, via act_noise). 
            if t > self.start_steps:
                a = self.get_action(self.o, self.act_noise)
            else:
                a = self.env.action_space.sample()

            # Step the env
            self.o2, self.r, self.d, _ = self.env.step(a)
            self.ep_ret += self.r
            self.ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if self.ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(self.o, self.a, self.r, self.o2, self.d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            self.o = self.o2

            # End of trajectory handling
            if d or (self.ep_len == self.max_ep_len):
                self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len)
                self.o, self.ep_ret, self.ep_len = self.env.reset(self.instances), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    self.batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=self.batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                self.epoch = (t+1) // self.steps_per_epoch

                # Save model
                if (self.epoch % self.save_freq == 0) or (self.epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                # Log info about epoch
                self.logger.log_tabular('Epoch', self.epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('QVals', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-self.start_time)
                self.logger.dump_tabular()

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='ddpg')
#     args = parser.parse_args()

#     from GCN_t.ddpg.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     a = DDPG(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#          ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#          gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#          logger_kwargs=logger_kwargs)
#     a.train()
