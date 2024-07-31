import threading
import psutil
import time
import heapq
import numpy as np


class CpuMemDetector(threading.Thread):
    """
    监控指定时间区间内，cpu和内存的使用情况
    """

    def __init__(self):
        super().__init__()
        self.end_detector = False
        self.min_mem = np.iinfo(np.int64).max
        self.max_mem = np.iinfo(np.int64).min
        self.max_cpus = []

    def run(self) -> None:
        while True:
            time.sleep(0.01)
            self.work()
            if self.end_detector:
                self.end_detector = False
                break

    def work(self):
        cpu_used_percent = round((psutil.cpu_percent()), 2)
        memory = psutil.virtual_memory()
        mem_used = memory.used // 1024
        heapq.heappush(self.max_cpus, -cpu_used_percent)
        self.max_cpus = self.max_cpus[:10]
        self.min_mem = min(self.min_mem, mem_used)
        self.max_mem = max(self.max_mem, mem_used)

    def end(self):
        self.end_detector = True

    def get_status(self):
        if len(self.max_cpus) == 0:
            # 太快没有记录上
            return 0, 0, 0
        else:
            return int(-1 * np.mean(self.max_cpus)), self.min_mem, self.max_mem
