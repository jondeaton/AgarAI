"""
File: training
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from dqn.qn import QN
from dqn import HyperParameters
from dqn.replay_buffer import ReplayBuffer
from dqn.replay_buffer import Transition

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

class Trainer(object):
    """ Trains a deep Q network (DQN) """

    def __init__(self, env, q, target_q, hyperams: HyperParameters, extractor=None):
        self.env = env
        self.replay_buffer = ReplayBuffer(hyperams.replay_buffer_capacity)

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
            os.makedirs(checkpoint_dir)

        self.q.train()
        episode_iterator = tqdm(range(num_episodes), unit="Episode")
        for ep in episode_iterator:
            ep_return = self.train_episode()

            episode_iterator.set_description("episode return: %.2f" % ep_return)
            tensorboard.log_scalar("train/EpisodeReturns(MP Relative)", ep_return, self.episodes)
            tensorboard.log_scalar("train/epsilon", self.epsilon, self.episodes)
            self.episodes += 1

            # save checkpoint
            should_checkpoint = checkpoint and (datetime.now() - self.last_save) > self.save_freq
            if should_checkpoint:
                logger.info(f"Check-pointing DQN Network...")
                torch.save(self.q, os.path.join(checkpoint_dir, "checkpoint"))
                self.last_save = datetime.now()

    def train_episode(self):
        """ train DQN for a single episode """
        total_returns = 0
        log = dict()

        self.env.reset()
        next_state = None
        done = False

        while not done:
            state = next_state

            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)

            total_returns += reward
            self.replay_buffer.push(state, action, next_state, reward)

            # only train for a step if the replay buffer is full
            if self.replay_buffer.full:
                self.train_step()
                self.gradient_steps += 1

                if self.gradient_steps % self.hyperams.target_update_freq == 0:
                    self.update_target_network()

            self.time_steps += 1

        return total_returns

    def train_step(self):
        """ Runs a single step of parameter optimization using a batch of experience
            examples from the replay buffer.
        """
        batch = self.replay_buffer.sample_pop(self.hyperams.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = self._to_tensor_batch(batch)

        Q_sa, _ = torch.max(self.q(state_batch), dim=1)
        target_Qspap, _ = torch.max(self.target_q(next_state_batch), dim=1)

        target = reward_batch + self.hyperams.gamma * target_Qspap

        loss = F.smooth_l1_loss(Q_sa, target)  # Hubert loss
        tensorboard.log_scalar("train/loss", float(loss), self.gradient_steps)
        self.optimize_q(loss)

    def optimize_q(self, loss: torch.Tensor):
        """ Performs a single step of optimization to minimise the loss """
        self.optimizer.zero_grad()

        if self.hyperams.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.hyperams.grad_clip_norm)

        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """ Updates the target network to have the parameters from the learned Q network """
        self.target_q.load_state_dict(self.q.state_dict())
        self.num_target_updates += 1

    def choose_action(self, state):
        """ Chooses an action to take in state 's' using epsilon-greedy policy """
        if state is None or torch.rand(1).item() <= self.epsilon:
            index = torch.randint(self.num_actions, (1,)).item()
        else:
            if self.extractor is not None:
                features = self.extractor.extract(state)
            else:
                features = state

            s_tensor = torch.from_numpy(features).type(torch.FloatTensor).to(self.device)
            qa = self.q(s_tensor)
            index = torch.argmax(qa).item()
        return self.to_action(index)

    def to_action(self, index):
        """ converts a raw action index into an action shape """
        indices = np.unravel_index(index, self.hyperams.action_shape)
        x = indices[0] / self.hyperams.action_shape[0]
        y = indices[1] / self.hyperams.action_shape[1]
        return x, y, indices[2]

    @property
    def epsilon(self):
        """ The current value of 'epsilon' for the e-greedy training policy """
        return self.hyperams.epsilon_base / pow(self.gradient_steps + 1, self.hyperams.epsilon_decay)

    def set_seed(self, seed):
        """ Sets random seeds for reproducibility """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

