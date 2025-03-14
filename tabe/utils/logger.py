import sys
import logging

_LOGGER_CACHE = {}  # {name: logging.Logger}

default_formatter = logging.Formatter(
    '%(asctime)s [%(filename)s(%(lineno)d)] - %(levelname)s: %(message)s'
#     '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s] - %(levelname)s: %(message)s'
)

_h_stdout = logging.StreamHandler(stream=sys.stdout)
_h_stdout.setFormatter(default_formatter)
_h_stdout.setLevel(logging.INFO)
default_handlers = [_h_stdout]

def get_logger(name, level="DEBUG", handlers=None, update=False, verbose=False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers 
    logger.propagate = False
    return logger

logger = get_logger("tabe_logger", handlers=default_handlers)
