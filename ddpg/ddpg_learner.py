from copy import copy
from functools import reduce
import functools

import numpy as np
import torch
from torch.optim import Adam
import ddpg.util as U
from ddpg.common.mpi_running_mean_std import RunningMeanStd
try:
    from mpi4py import MPI
except ImportError:
    MPI = None



def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = torch.mean(x, dim=axis, keepdim=keepdims)
    devs_squared = torch.square(x - m)
    return torch.mean(devs_squared, dim=axis, keepdim=keepdims)

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

def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.vars) == len(perturbed_actor.vars)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    updates = []
    for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
        if var in actor.perturbable_vars:
            # logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            perturbed_var = var + torch.normal(torch.zeros(var.shape),std=param_noise_stddev)
            updates.append(perturbed_var)
        else:
            # logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            perturbed_var = var
            updates.append(perturbed_var)
    assert len(updates) == len(actor.vars)
    return updates

def samp(act):

    act = np.random.binomial(1, act)
    act = np.where(act > 0.5, act, 0.)  
    act = np.where(act == 0., act, 1.) 

    return act

def entry_stop_gradients(target, mask):
    mask_h = torch.abs(mask-1)
    return (mask_h * target).detach() + mask * target


class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-np.inf, np.inf), action_range=(0.2, 0.8), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):
        # Inputs.
        # self.obs0 = 
        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        # if self.normalize_observations:
        #     with tf.variable_scope('obs_rms'):
        #         self.obs_rms = RunningMeanStd(shape=observation_shape)
        # else:
        #     self.obs_rms = None
        if self.normalize_observations == True:
            self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None
        normalized_obs0 = torch.clip(normalize(self.obs0, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        normalized_obs1 = torch.clip(normalize(self.obs1, self.obs_rms),
            self.observation_range[0], self.observation_range[1])
        
        # Return normalization.
        if self.normalize_returns == True:
            self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        target_actor = copy(actor)
        self.target_actor = target_actor 
        target_critic = copy(critic)
        self.target_critic = target_critic

        # Create networks and core TF parts that are shared across setup parts.
        self.actor_tf = actor(normalized_obs0)

        self.actor_tf_next = actor(normalized_obs1)   ###action pros. for next state for entropy computation
        
        self.normalized_critic_tf = critic(normalized_obs0, self.actions)
        self.critic_tf = denormalize(torch.clip(self.normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = critic(normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(torch.clip(self.normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        self.target_act = target_actor(normalized_obs1)
        Q_obs1 = denormalize(target_critic(normalized_obs1, self.next_actions), self.ret_rms)
        self.target_Q = self.rewards + gamma * Q_obs1

        self.Q_obs0 = denormalize(critic(normalized_obs0, self.actions), self.ret_rms)        #QA2C

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise(normalized_obs0)
        self.setup_actor_optimizer()
        self.setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self.setup_popart()
        self.setup_stats()
        self.setup_target_network_updates()

        self.initial_state = None # recurrent architectures not supported yet

        def setup_target_network_updates(self):
            actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
            critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
            self.target_init_updates = [actor_init_updates, critic_init_updates]
            self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

        def setup_param_noise(self, normalized_obs0):
            assert self.param_noise is not None

            # Configure perturbed actor.
            param_noise_actor = copy(self.actor)
            # param_noise_actor.name = 'param_noise_actor'
            self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
            # logger.info('setting up param noise')
            self.perturb_policy_ops = get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

            # Configure separate copy for stddev adoption.
            adaptive_param_noise_actor = copy(self.actor)
            adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
            adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
            self.perturb_adaptive_policy_ops = get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
            self.adaptive_policy_distance = torch.sqrt(torch.mean(torch.square(self.actor_tf - adaptive_actor_tf)))
        
        def setup_actor_optimizer(self):
            # logger.info('setting up actor optimizer')
            self.choice = torch.cat([1-self.actor_tf, self.actor_tf], 2)

            self.choice1 = torch.reshape(self.choice,[-1,2]) 

            self.indice = torch.concat([torch.expand_dims(torch.arange(125000), -1), torch.reshape(self.actions.int(),[-1,1])], -1)

            self.decision = U.gather_nd(self.choice1,self.indice)

            self.decision1 = torch.reshape(self.decision,[-1,1000,1])

            self.actor_loss = -torch.mean(torch.sum(torch.log(self.decision1),1)*self.Q0)

            actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
            actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
            # logger.info('  actor shapes: {}'.format(actor_shapes))
            # logger.info('  actor params: {}'.format(actor_nb_params))
            self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)

            self.actor_optimizer = Adam(params=self.actor.trainable_vars,lr=self.actor_lr)

        def setup_critic_optimizer(self):
            # logger.info('setting up critic optimizer')
            normalized_critic_target_tf = torch.clip(normalize(self.critic_target, self.ret_rms), self.return_range[0], self.return_range[1])
            self.critic_loss = torch.mean(torch.square(self.normalized_critic_tf - normalized_critic_target_tf))
            if self.critic_l2_reg > 0.:
                critic_reg_vars = [var for var in self.critic.trainable_vars if var.name.endswith('/w:0') and 'output' not in var.name]
                # for var in critic_reg_vars:
                    # logger.info('  regularizing: {}'.format(var.name))
                # logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
                critic_reg = tc.layers.apply_regularization(
                    tc.layers.l2_regularizer(self.critic_l2_reg),
                    weights_list=critic_reg_vars
                )
                self.critic_loss += critic_reg
            critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
            critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
            # logger.info('  critic shapes: {}'.format(critic_shapes))
            # logger.info('  critic params: {}'.format(critic_nb_params))
            self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
            self.critic_optimizer = Adam(params=self.critic.trainable_vars,lr=self.critic_lr)

        pass
