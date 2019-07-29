"""
File: Coordinator
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)

Inspired by: https://github.com/MG2033/A2C

"""

import numpy as np
from enum import Enum

import multiprocessing as mp
from multiprocessing import Pipe, Process


class RemoteCommand(Enum):
    step = 1
    reset = 2
    close = 3


def worker_task(pipe: mp.connection.PipeConnection, get_env):
    env = get_env()
    while True:
        command, data = pipe.recv()
        if command == RemoteCommand.step:
            step_data = env.setp(data)
            pipe.send(step_data)
        elif command == RemoteCommand.reset:
            ob = env.reset()
            pipe.send(ob)
        elif command == RemoteCommand.close:
            pipe.close()
            return
        else:
            raise ValueError(command)


class Coordinator:
    def __init__(self, get_env, num_workers):
        self.num_workers = num_workers

        worker_pipes = [Pipe() for _ in range(self.num_workers)]
        self.pipes = [pipe for _, pipe in worker_pipes]
        self.workers = [Process(target=worker_task,
                                args=(pipe, get_env)) for pipe, _ in worker_pipes]

        for worker in self.workers:
            worker.start()

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            package = RemoteCommand.step, action
            pipe.send(package)

        results = [pipe.recv() for pipe in self.pipes]
        obs, rs, dones, infos = zip(*results)
        return np.hstack(obs), np.hstack(rs), np.hstack(dones), infos

    def reset(self):
        for pipe in self.pipes:
            pipe.send(RemoteCommand.reset)

        obs = [pipe.recv() for pipe in self.pipes]
        return np.hstack(obs)

    def close(self):
        for pipe in self.pipes:
            pipe.send(RemoteCommand.close)
        for worker in self.workers:
            worker.join()