"""
File: async_coordinator
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

import gym

import threading
from collections import deque

from a2c.rollout import Rollout
from a2c.remote_environment import worker_task, RemoteCommand, RemoteEnvironment


def get_rollout(env: RemoteEnvironment, episode_length, initialize, get_actions, to_action):
    rollout = Rollout()

    observations = env.reset()

    dones = None

    carry = initialize()

    for t in range(episode_length):
        if dones is None or not all(dones):
            actions, values, carry = get_actions(observations, carry)

            act_in = list(map(to_action, actions))
            next_obs, rewards, dones, _ = env.step(act_in)

            rollout.record(observations, actions, rewards, values, dones)
            observations = next_obs

    return rollout


def thread_task(get_env, episode_length, initialize, get_actions, to_action,
                cv: threading.Condition, queue: deque):

    with RemoteEnvironment(get_env) as env:

        while True:
            rollout = get_rollout(env, episode_length, initialize, get_actions, to_action)
            with cv:
                queue.append(rollout)
                cv.notify()


class AsyncCoordinator:

    def __init__(self, num_envs, get_env, episode_length, initialize, get_actions, to_action):

        self.num_envs = num_envs
        self.get_env = get_env
        self.get_actions = get_actions

        self.cv = threading.Condition()
        self.rollout_queue = deque()

        self.args = (get_env,
                     episode_length, initialize, get_actions, to_action,
                     self.cv, self.rollout_queue)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def start(self):
        self.threads = []
        for _ in range(self.num_envs):
            thread = threading.Thread(target=thread_task, args=self.args)
            thread.start()
            self.threads.append(thread)

    def await_rollout(self):
        """ waits until there is a rollout available """
        with self.cv:
            while not self.rollout_queue:
                self.cv.wait()
            return self.rollout_queue.popleft()

    def close(self):
        for thread in self.threads:
            thread.join()
        self.threads.clear()