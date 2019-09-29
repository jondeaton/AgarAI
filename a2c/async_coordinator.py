"""
File: async_coordinator
Date: 9/21/19 
Author: Jon Deaton (jonpauldeaton@gmail.com)
"""

from multiprocessing import Process, Queue, Condition, Semaphore


class AsyncCoordinator:
    """ manages a collection of worker processes that
    produce data to be consumed by the client of this class.
    """
    def __init__(self, num_workers, worker_target, args):
        """ Construct
        :param num_workers: the number of worker processes
        :param worker_target: function for each worker process to execute
        :param args: the aforementioned additional
        arguments to be passed to each worker
        """
        self.num_workers = num_workers
        self.worker_target = worker_target
        self.args = args

        self.queue = None
        self.sema = None
        self._workers = None

    def open(self):
        """ creates the collection of managed worker processes """
        self.queue = Queue()
        self.sema = Semaphore(0)

        self._workers = []
        for wid in range(self.num_workers):
            worker = Process(target=self.worker_target,
                             args=(wid, self.queue, self.sema) + self.args)
            worker.start()
            self._workers.append(worker)

    def close(self):
        """ destroys the worker processes """
        del self.queue
        for worker in self._workers:
            worker.terminate()
            worker.join()

    def start(self):
        """ signals all worker processes to begin """
        for _ in range(self.num_workers):
            self.sema.release()

    def pop(self):
        """ blocks until there is a datum in the queue
        produced by a worker, then removes and returns it.
        """
        if self.queue is None:
            raise Exception()
        return self.queue.get()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
