"""
File: async_coordinator
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""


from a2c.rollout import Rollout
from multiprocessing import Process, Queue, Value


def worker_task(queue: Queue, get_rollout, get_env, model_directory):

    env = get_env()

    import tensorflow as tf

    while True:

        # get the latest model
        model = tf.keras.models.load_model(model_directory)
        rollout = get_rollout(model, env)
        queue.put(rollout)


class AsyncCoordinator:
    """ asynchronous remote  """
    def __init__(self, num_envs, model_directory, get_env, get_rollout):
        self.num_envs = num_envs
        self.model_directory = model_directory
        self.get_rollout = get_rollout
        self.get_env = get_env

        self.queue = None

    def __enter__(self):
        self.queue = Queue()
        self.workers = []
        for w in range(self.num_envs):

            args = (self.queue, self.get_rollout, self.get_env, self.model_directory)

            worker = Process(target=worker_task, args=args)
            worker.start()
            self.workers.append(worker)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.queue

    def await_rollout(self):
        if self.queue is None:
            raise Exception
        return self.queue.get()


