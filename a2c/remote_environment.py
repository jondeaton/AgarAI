"""
File: worker_process
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""


from enum import Enum
from multiprocessing import Pipe, Process


class RemoteCommand(Enum):
    step = 1
    reset = 2
    close = 3
    observation_space = 4
    action_space = 5


class RemoteEnvironment:
    """ encapsulates a multi-agent environment in a remote process """

    def __init__(self, get_env):
        self.get_env = get_env

    def __enter__(self):
        worker_pipe, self._pipe = Pipe()
        self._worker = Process(target=worker_task, args=(worker_pipe, self.get_env))
        self._worker.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self._pipe.send(RemoteCommand.close)
        self._worker.join()

    def reset(self):
        self._pipe.send(RemoteCommand.reset)
        obs = self._pipe.recv()
        return obs

    def step(self, actions):
        self._pipe.send((RemoteCommand.step, actions))
        obs, rewards, dones, info = self._pipe.recv()
        return obs, rewards, dones, info

    def observation_space(self):
        self._pipe.send(RemoteCommand.observation_space)
        return self._pipe.recv()

    def action_space(self):
        self._pipe.send(RemoteCommand.action_space)
        return self._pipe.recv()


def worker_task(pipe, get_env):
    env = get_env()

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
