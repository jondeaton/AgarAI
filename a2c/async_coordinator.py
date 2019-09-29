"""
File: async_coordinator
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""



from a2c.rollout import Rollout
from multiprocessing import Process, Queue, Value


def worker_task(wid, queue: Queue, get_rollout, make_model, get_env, model_directory):
    """ the task that each worker process performs: gather the complete
     roll-out of an episode using the latest model and send it back to the
     master process. """
    env = get_env()
    model = make_model()

    while True:
        model.load_weights(model_directory)

        print(f"Worker {wid} starting episode...")
        rollout: Rollout = get_rollout(model, env)
        print(f"Worker {wid} episode finished")

        queue.put(rollout)


class AsyncCoordinator:
    """ asynchronous remote  """
    def __init__(self, num_envs, model_directory, make_model, get_env, get_rollout):
        self.num_envs = num_envs
        self.model_directory = model_directory
        self.get_rollout = get_rollout
        self.make_model = make_model
        self.get_env = get_env

        self.queue = None

    def __enter__(self):
        self.queue = Queue()
        self.workers = []
        for w in range(self.num_envs):

            args = (w, self.queue, self.get_rollout, self.make_model, self.get_env, self.model_directory)

            worker = Process(target=worker_task, args=args)
            worker.start()
            self.workers.append(worker)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.queue
        for worker in self.workers:
            worker.terminate()
            worker.join()

    def await_rollout(self):
        if self.queue is None:
            raise Exception
        return self.queue.get()


