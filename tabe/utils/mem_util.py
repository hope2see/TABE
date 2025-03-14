import os
import psutil
import tracemalloc
from tabe.utils.logger import logger


class MemUtil(object):
    _tracemalloc_started = False 

    def __init__(self, rss_mem=True, python_mem=True):
        self.rss_mem = rss_mem
        self.python_mem = python_mem
        self.rss_mem_init = None
        self.rss_mem_prev = None
        self.python_mem_init = None
        self.python_mem_prev = None

    def print_rss_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        current = mem_info.rss / (1024 ** 2) # to MegaByte
        if self.rss_mem_init is None:
            self.rss_mem_init = current
            self.rss_mem_prev = current
        logger.info(f"RSS Mem Usage: cur={current:.2f} incr={current - self.rss_mem_prev:.2f} "
                f"acc_incr={current - self.rss_mem_init:.2f} MB")
        self.rss_mem_prev = current

    def start_python_memory_tracking(self):
        if MemUtil._tracemalloc_started:
            logger.warning("Warning: start_python_memory_tracking() is called before stop_python_memory_tracking()")
        else:
            tracemalloc.start()
            MemUtil._tracemalloc_started = True

    def stop_python_memory_tracking(self):
        if MemUtil._tracemalloc_started:
            tracemalloc.stop()
            MemUtil._tracemalloc_started = False
        else:
            logger.warning("Warning: stop_python_memory_tracking() is called before start_python_memory_tracking()")

    def print_python_memory_usage(self):
        current, peak = tracemalloc.get_traced_memory()
        current = current / (1024 ** 2) # to MegaByte
        peak = peak / (1024 ** 2) # to MegaByte
        if self.python_mem_init is None:
            self.python_mem_init = current
            self.python_mem_prev = current
        logger.info(f"Python Mem Usage: cur={current:.2f} incr={current - self.python_mem_prev:.2f} "
                f"acc_incr={current - self.python_mem_init:.2f} peak={peak:.2f} MB")
        self.python_mem_prev = current

    def print_memory_usage(self, rss=None, python=None):
        rss = self.rss_mem if rss is None else rss
        python = self.python_mem if python is None else python
        if rss:
            self.print_rss_memory_usage()
        if python:           
            self.print_python_memory_usage()
