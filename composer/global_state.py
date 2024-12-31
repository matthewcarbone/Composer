import sys

_this = sys.modules[__name__]


def set_memory_dir(target=None):
    _this.memory_dir = target


def get_memory_dir():
    return _this.memory_dir


set_memory_dir(None)
