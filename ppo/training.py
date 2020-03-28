"""
File: training
Date: 2/1/20 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

from a2c.hyperparameters import HyperParameters
from a2c.rollout import Rollout, transpose_batch
from a2c.async_coordinator import AsyncCoordinator

import os
from tqdm import tqdm
import numpy as np
import random
import logging
logger = logging.getLogger("root")

import multiprocessing as mp

import jax.numpy as jnp
from jax import jit, random

# Limit operations to single thread when on CPU.
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def get_rollout(model, env, agents_per_env, episode_length, to_action, record=True, progress_bar=False) -> Rollout:
    """ performs a roll-out """
    rollout = Rollout() if record else None

    observations = env.reset()
    dones = [False] * agents_per_env

    step_iterator = tqdm(range(episode_length)) if progress_bar else range(episode_length)
    for _ in step_iterator:
        if all(dones): break

        action_logits, values_estimate = model(observations)

        # Choose actions.
        action_indices = random.categorical(key, softmax(action_logits))
        values = list(values_estimate)
        
        actions = [to_action(i) for i in action_indices]
        next_obs, rewards, next_dones, _ = env.step(actions)

        if record:
            rollout.record(observations, action_indices, rewards, values, dones)
        
        dones = next_dones
        observations = next_obs

    return rollout


def _update_with_rollout(model, rollout_batch):
    """ updates the network using a roll-out """

    from a2c.losses import a2c_loss, get_loss_variables, batch_loss_variables

    loss_vars = get_loss_variables(rollout_batch, self.hyperams.gamma, model.recurrent)

    if self.hyperams.batch:
        for loss_vars_batch in batch_loss_variables(loss_vars, self.hyperams.batch_size):
            losses, grads = a2c_loss(model, self.hyperams.entropy_weight, *loss_vars_batch)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    else:
        losses, grads = a2c_loss(model, self.hyperams.entropy_weight, *loss_vars)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses


def worker_fn(wid, model, optimizer):
    rollout = get_rollout(model)
    gradients = compute_gradients(model, rollout)


class Trainer:

    def __init__(self, get_env, hyperams: HyperParameters, to_action, test_env=None, training_dir=None):
        self.get_env = get_env
        self.hyperams = hyperams
        self.to_action = to_action
        self.test_env = test_env
        self.num_envs = hyperams.num_envs
        self.training_dir = training_dir

    def train(self, asynchronous=False):
        """ trains a model """
        if asynchronous:
            self._train_async()
        else:
            self._train_sync()

    def _train_ppo(self):
        env = self.get_env()

        input_shape = (None,) + env.observation_space.shape
        model = make_model(self.hyperams.architecture,
                           self.hyperams.encoder_class,
                           input_shape,
                           self.hyperams.action_shape)

        self.optimizer = tf.keras.optimizers.Adam(lr=self.hyperams.learning_rate, clipnorm=1.0)

        summary_writer = None
        if self.training_dir is not None:
            model_directory = os.path.join(self.training_dir, "model")
            summary_writer = tf.summary.create_file_writer(self.training_dir)

        for ep in range(self.hyperams.num_episodes):
            logger.info(f"Episode {ep}")
            episode_length = self.hyperams.episode_length
            rollout = get_rollout(model, env,
                                  self.hyperams.agents_per_env,
                                  episode_length,
                                  self.to_action)

            losses = self._update_with_rollout(model, rollout.as_batch())
            self._log_rollout(summary_writer, ep, rollout.as_batch(), losses)

            if self.training_dir is not None and ep % self.hyperams.save_frequency == 0:
                logger.info("Saving model checkpoint.")
                model.save_weights(model_directory)


    def _train_sync(self):
        raise NotImplementedError()

    def _train_async(self):
        """ trains a model asynchronously """

        model = get_model()

        queue = mp.Queue()

        logger.info(f"Spawning {self.hyperams.num_envs} workers...")
        workers = []
        with mp.get_context('spawn') as context:
            for i in self.hyperams.num_envs:
                args = i, model
                process = context.Process(target=worker_fn, args=args)
                workers.append(process)
        logger.info(f"Workers spawned...")

        model_directory = os.path.join(self.training_dir, "model")

        args = self.get_env, model_directory, self.to_action, self.hyperams
        coordinator = AsyncCoordinator(self.hyperams.num_envs,
                                       worker_target, args)

        with coordinator:
            import tensorflow as tf
            from a2c.eager_models import make_model

            summary_writer = tf.summary.create_file_writer(self.training_dir)

            input_shape = (None,) + self.get_env().observation_space.shape
            model = make_model(self.hyperams.architecture,
                               self.hyperams.encoder_class,
                               input_shape,
                               self.hyperams.action_shape)

            self.optimizer = tf.keras.optimizers.Adam(lr=self.hyperams.learning_rate, clipnorm=1.0)

            logger.info("Saving initial model...")
            model.save_weights(model_directory)
            logger.info("Initial model saved.")

            coordinator.start()  # start the worker processes

            for episode in range(self.hyperams.num_episodes):
                rollout_batch = coordinator.pop()

                losses = self._update_with_rollout(model, rollout_batch)
                self._log_rollout(summary_writer, episode, rollout_batch, losses=losses)

                model.save_weights(model_directory)



    def _log_rollout(self, summary_writer, episode, rollout_batch, losses=None):
        """ logs the performance of the roll-out """
        episode_length = len(rollout_batch[2][0])
        logger.info(f"Episode {episode}, length: {episode_length}")
        if losses is not None:
            logger.info(f"Actor loss: {losses[0]:.3f}, Critic loss: {losses[1]:.3f}")

        returns = []
        max_masses = []
        average_masses = []
        efficiencies = []

        for rewards in rollout_batch[2]:
            episode_return = rewards.sum()
            mass = 10 + rewards.cumsum()
            eff = get_efficiency(rewards, episode_length, self.hyperams)

            returns.append(episode_return)
            max_masses.append(mass.max())
            average_masses.append(mass.mean())
            efficiencies.append(eff)

        print(f"Average Ep Return:\t{np.mean(returns):.2f}")
        print(f"Average Max mass:\t{np.mean(max_masses):.2f}")
        print(f"Average Avg mass:\t{np.mean(average_masses):.2f}")
        print(f"Average efficiency:\t{np.mean(efficiencies):.2f}")

        if summary_writer is not None:  # in debug mode theres no directory to write to
            import tensorflow as tf
            with summary_writer.as_default():
                tf.summary.scalar('train/efficiency', np.mean(efficiencies), step=episode)
                if losses is not None:
                    tf.summary.scalar('loss/actor', losses[0], step=episode)
                    tf.summary.scalar('loss/critic', losses[1], step=episode)

    def _test(self, model, summary_writer=None, episode_length=None):
        logger.info(f"Testing performance...")
        episode_length = episode_length or self.hyperams.episode_length
        rollout = get_rollout(model, self.test_env, self.hyperams.agents_per_env,
                              episode_length, self.to_action)

        # todo: pass the real summary writer to log results to TensorBoard.
        self._log_rollout(summary_writer, "test", rollout.as_batch(), episode_length)


def get_efficiency(rewards, episode_length, hyperams):
    """ calculates the "agario mass efficiency", which is a quantity that i invented lol
    It is supposed to capture the rate at which mass was accumulated relative to
    the density of pellets in the arena. In this way, performance can be
    compared across episodes of different lengths, arenas of different sizes
    and with different numbers of pellets.
    """
    G = rewards.sum()
    pellet_density = hyperams.num_pellets / pow(hyperams.arena_size, 2)
    efficiency = G / (episode_length * pellet_density)
    return efficiency