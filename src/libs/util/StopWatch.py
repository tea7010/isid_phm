import time


class StopWatch:
    def __init__(self):
        self.st = time.time()
        self.t = self.st

    def ex_time(self, process='Process time: '):
        print(process, time.time() - self.t)
        self.t = time.time()

    def total_time(self):
        self.ed = time.time()
        print('Total time: ', self.ed - self.st)
