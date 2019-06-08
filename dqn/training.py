"""
File: training
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from dqn import HyperParameters

from dqn.replay_memory import ReplayMemory

from datetime import datetime, timedelta
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import os
from log import tensorboard
import logging
logger = logging.getLogger('root')

from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

from typing import List, Tuple

class Trainer(object):
    """ Trains a deep Q network (DQN) """

    def __init__(self, env, q, target_q, hyperams: HyperParameters, extractor=None):
        self.env = env
        self.replay_memory = ReplayMemory(hyperams.replay_memory_capacity)

        self.device = q.device

        self.q = q
        self.target_q = target_q

        self.hyperams = hyperams
        self.extractor = extractor

        self.num_actions = int(np.prod(self.hyperams.action_shape))

        self.optimizer = optim.Adam(q.parameters(),
                                    lr=self.hyperams.lr,
                                    betas=self.hyperams.adam_betas,
                                    eps=self.hyperams.adam_eps)

        self.time_steps = 0
        self.episodes = 0

        self.gradient_steps = 0
        self.num_target_updates = 0

        self.last_save = datetime.now()
        self.save_freq = timedelta(minutes=5)

        self.set_seed(hyperams.seed)


    def train(self, num_episodes=None, training_dir=None):
        """ trains the DQN for a single epoch consisting of `num_episodes` """
        num_episodes = num_episodes or self.hyperams.num_episodes

        checkpoint = training_dir is not None
        if checkpoint:
            checkpoint_dir = os.path.join(training_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.q.train()
        episode_iterator = tqdm(range(num_episodes), unit="Episode")
        for ep in episode_iterator:
            ep_return = self.train_episode()

            episode_iterator.set_description("episode return: %.2f" % ep_return)
            tensorboard.log_scalar("train/EpisodeReturns", ep_return, self.episodes)
            tensorboard.log_scalar("train/epsilon", self.epsilon, self.episodes)
            self.episodes += 1

            # save checkpoint
            should_checkpoint = checkpoint and (datetime.now() - self.last_save) > self.save_freq
            if should_checkpoint:
                torch.save(self.q, os.path.join(checkpoint_dir, "checkpoint"))
                self.last_save = datetime.now()

    def train_episode(self):
        """ train DQN for a single episode """
        total_returns = 0
        log = dict()

        self.env.reset()

        next_state_fts = None
        for i in range(self.hyperams.episode_length):
            state_fts = next_state_fts

            action_index = self.choose_action(state_fts)
            action = self.to_action(action_index)
            next_state, reward, done, info = self.env.step(action)

            self.env.render()

            next_state_fts = self.to_features(next_state)
            total_returns += reward

            if state_fts is not None:
                transition = Transition(state_fts, action_index, next_state_fts, reward)
                errors = self.get_errors([transition])
                self.replay_memory.push(errors[0], transition)

            # only train for a step if the replay buffer is full
            if self.time_steps % self.hyperams.lean_freq == 0 and self.replay_memory.full():
                self.train_step()
                self.gradient_steps += 1

                if self.gradient_steps % self.hyperams.target_update_freq == 0:
                    self.update_target_network()

            self.time_steps += 1
            if done: break

        return total_returns

    def train_step(self):
        """ Runs a single step of parameter optimization using a batch of experience
            examples from the replay buffer.
        """
        # sample an experience batch from replay buffer
        batch, indexes, _ = self.replay_memory.sample(self.hyperams.batch_size)
        sars_batches = self.to_tensor_batch(batch)
        state_batch, action_batch, reward_batch, _, next_state_batch = sars_batches

        # the estimate for the quality of taking those actions in those states
        Q_sa = self.get_Qsa(self.q, state_batch, action_batch)

        # the target of these estimates, using one-step backup
        target = self.get_target(sars_batches)

        loss = F.smooth_l1_loss(Q_sa, target)  # Huber loss
        tensorboard.log_scalar("train/loss", float(loss), self.gradient_steps)
        tensorboard.log_scalar("train/Qsa", float(torch.mean(Q_sa)), self.gradient_steps)

        # update the new errors in the replay memory
        errors = torch.abs(Q_sa - target).cpu().data.numpy()
        for i in range(self.hyperams.batch_size):
            self.replay_memory.update(indexes[i], errors[i])

        self.optimize_q(loss)

    def get_target(self, sars_batches: Tuple):
        _, _, reward_batch, non_final_mask, next_state_batch = sars_batches
        Q_tgt = torch.zeros_like(reward_batch, device=self.device)

        if self.hyperams.double_dqn:
            # double DQN: choose action using online network
            ap = torch.argmax(self.q(next_state_batch), dim=1)
            # but use "stationary" target network to evaluate V(s')
            Q_tgt[non_final_mask] = self.get_Qsa(self.target_q, next_state_batch, ap)
        else:
            # vanilla DQN: just use target network maximization
            Q_tgt[non_final_mask], _ = torch.max(self.target_q(next_state_batch), dim=1)

        target = reward_batch + self.hyperams.gamma * Q_tgt
        return target

    def get_Qsa(self, q, state_batch, action_batch):
        Q_s = q(state_batch) # values for all actions
        # now, select from those, the actions specified in action_batch
        return torch.gather(Q_s, 1, action_batch.unsqueeze(dim=1)).squeeze()

    def get_errors(self, transition_batch: List[Transition]):
        sars_batches = self.to_tensor_batch(transition_batch)
        state_batch, action_batch, reward_batch, _, next_state_batch = sars_batches

        Q_sa = self.get_Qsa(self.q, state_batch, action_batch)
        target = self.get_target(sars_batches)
        errors = torch.abs(Q_sa - target)
        return errors.cpu().data.numpy()

    def optimize_q(self, loss: torch.Tensor):
        """ Performs a single step of optimization to minimise the loss """
        self.optimizer.zero_grad()

        if self.hyperams.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.hyperams.grad_clip_norm)

        loss.backward()

        for param in self.q.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_target_network(self):
        """ Updates the target network to have the parameters from the learned Q network """
        self.target_q.load_state_dict(self.q.state_dict())
        self.num_target_updates += 1

    def choose_action(self, features):
        """ Chooses an action to take in state 's' using epsilon-greedy policy """
        if features is None or random.random() <= self.epsilon:
            index = random.randint(0, self.num_actions - 1)
        else:

            def fix_features(ft):
                return torch.from_numpy(ft).unsqueeze(dim=0).type(torch.FloatTensor).to(self.device)

            s_tensor = fix_features(features)
            qa = self.q(s_tensor)

            index = torch.argmax(qa).item()
        return index

    def to_action(self, index):
        """ converts a raw action index into an action shape """
        indices = np.unravel_index(index, self.hyperams.action_shape)
        theta = 2 * np.pi * indices[0] / self.hyperams.action_shape[0]
        mag = 1 - indices[1] / self.hyperams.action_shape[1]
        act = indices[2]
        x = np.cos(theta) * mag
        y = np.sin(theta) * mag
        return x, y, act

    def to_tensor_batch(self, transition_batch):
        """ converts a batch (i.e. simple list) of Transition objects
            into a tuple of tensors for the batch of states, actions,
            rewards, and next states. Yes... this function is poorly written.
        """
        batch = Transition(*zip(*transition_batch))

        def to_tensor(array_list):
            """ converts a list of numpy arrays into a single tensor """
            tensors = tuple(torch.from_numpy(np.array(a)) for a in array_list if a is not None)
            return torch.stack(tensors, dim=0).type(torch.FloatTensor).to(self.device)

        feature_batch = to_tensor(batch.state)
        action_batch = to_tensor(batch.action).to(torch.long)
        reward_batch = to_tensor(batch.reward)
        next_feature_batch = to_tensor(batch.next_state)

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=self.device, dtype=torch.uint8)

        return feature_batch, action_batch, reward_batch, non_final_mask, next_feature_batch

    def to_features(self, observation):
        if self.extractor is None or observation is None:
            return observation
        else:
            return self.extractor(observation)

    @property
    def epsilon(self):
        """ The current value of 'epsilon' for the e-greedy training policy """
        r = np.exp(- self.hyperams.epsilon_decay * self.gradient_steps)
        diff = self.hyperams.epsilon_base - self.hyperams.epsilon_end
        return self.hyperams.epsilon_end + r * diff

    def set_seed(self, seed):
        """ Sets random seeds for reproducibility """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

