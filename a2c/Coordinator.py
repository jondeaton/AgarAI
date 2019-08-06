"""
File: Coordinator
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)

Inspired by: https://github.com/MG2033/A2C

"""
import gym
import numpy as np
from enum import Enum
import logging

from typing import List

import multiprocessing as mp
from multiprocessing import Pipe, Process

logger = logging.getLogger()


class RemoteCommand(Enum):
    step = 1
    reset = 2
    close = 3
    observation_space = 4
    action_space = 5


def worker_task(pipe, get_env):
    env: gym.Env = get_env()

    while True:
        try:
            msg = pipe.recv()
        except (KeyboardInterrupt, EOFError):
            return

        if type(msg) is tuple:
            command, data = msg
        else:
            command = msg

        if command == RemoteCommand.step:
            step_data = env.step(data)
            pipe.send(step_data)
        elif command == RemoteCommand.reset:
            ob = env.reset()
            pipe.send(ob)
        elif command == RemoteCommand.close:
            pipe.close()
            return
        elif command == RemoteCommand.observation_space:
            pipe.send(env.observation_space)
        elif command == RemoteCommand.action_space:
            pipe.send(env.action_space)
        else:
            raise ValueError(command)


class Coordinator:
    def __init__(self, get_env, num_workers):
        self.get_env = get_env
        self.num_workers = num_workers

        self.pipes = None
        self.workers = None

        self.dones = [False] * num_workers

    def open(self):
        worker_pipes = [Pipe() for _ in range(self.num_workers)]
        self.pipes = [pipe for _, pipe in worker_pipes]
        self.workers = [Process(target=worker_task,
                                args=(pipe, self.get_env)) for pipe, _ in worker_pipes]

        for worker in self.workers:
            worker.start()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def step(self, actions):
        for i in range(self.num_workers):
            if not self.dones[i]:
                msg = RemoteCommand.step, actions[i]
                self.pipes[i].send(msg)

        obs = list()
        rs = list()
        infos = list()
        for i in range(self.num_workers):
            if not self.dones[i]:
                o, r, done, info = self.pipes[i].recv()
                obs.append(o)
                rs.append(r)
                self.dones[i] = done
                infos.append(info)
            else:
                obs.append(None)
                rs.append(None)
                infos.append(None)

        return obs, rs, self.dones.copy(), infos

    def reset(self):
        for pipe in self.pipes:
            pipe.send(RemoteCommand.reset)

        self.dones = [False] * self.num_workers
        obs = [pipe.recv() for pipe in self.pipes]
        return obs

    def close(self):
        for pipe in self.pipes:
            pipe.send(RemoteCommand.close)

        for worker in self.workers:
            worker.join()

    def observation_space(self):
        self.pipes[0].send(RemoteCommand.observation_space)
        return self.pipes[0].recv()

    def action_space(self):
        self.pipes[0].send(RemoteCommand.action_space)
        return self.pipes[0].recv()

    def send(self, pipe, *args):
        try:
            pipe.send(args)
        except BrokenPipeError:
            pass

    def recv(self, pipe):
        try:
            return pipe.recv()
        except EOFError:
            return None