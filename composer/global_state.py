import sys

_this = sys.modules[__name__]


# Memory directory, which is required in the global context of some modules


def set_memory_dir(target=None):
    _this.memory_dir = target  # type: ignore


def get_memory_dir():
    return _this.memory_dir


# Verbosity


def set_verbosity(verbosity):
    _this.verbose = verbosity  # type: ignore


def get_verbosity():
    return _this.verbose


# Defaults

set_memory_dir(None)
set_verbosity(False)
