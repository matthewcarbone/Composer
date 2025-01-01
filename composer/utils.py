import hashlib
from time import perf_counter


class Timer:
    def __enter__(self):
        self._time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self._time = perf_counter() - self._time

    @property
    def dt(self):
        return self._time


def get_file_hash(file_path):
    """Compute and return the SHA-256 hash of the given file."""

    hasher = hashlib.sha256()
    block_size = 8192  # read in chunks of 8 KB
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()
