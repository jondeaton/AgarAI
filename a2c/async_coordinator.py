"""
File: async_coordinator
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

from multiprocessing import Process, Queue, Condition, Semaphore

class AsyncCoordinator:
    """ asynchronous remote  """
    def __init__(self, num_workers, worker_target, args):
        self.num_workers = num_workers
        self.worker_target = worker_target
        self.args = args

        self.queue = None
        self.sema = None
        self._workers = None

    def open(self):
        self.queue = Queue()
        self.sema = Semaphore(0)

        self._workers = []
        for wid in range(self.num_workers):
            worker = Process(target=self.worker_target,
                             args=(wid, self.queue, self.sema) + self.args)
            worker.start()
            self._workers.append(worker)

    def close(self):
        del self.queue
        for worker in self._workers:
            worker.terminate()
            worker.join()

    def start(self):
        for _ in range(self.num_workers):
            self.sema.release()

    def pop(self):
        if self.queue is None:
            raise Exception()
        return self.queue.get()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
