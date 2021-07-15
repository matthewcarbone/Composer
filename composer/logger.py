"""
Authors
-------
Cole Miles
    Cornell University, Department of Physics
Matthew R. Carbone
    Brookhaven National Laboratory, Computational Science Initiative
"""

import datetime
import json
from pathlib import Path

import numpy as np


class Logger:
    """Class responsible for logging all relevant information to disk and the
    console."""

    def __init__(self, root, logger_id=None):
        """
        Parameters
        ----------
        root : str
            The root directory. All logged information will be saved there.
        """

        self.root = Path(root)
        if logger_id is not None:
            self.root = self.root / Path(logger_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            'loss': str(self.root / Path("loss.txt")),
            'meta': str(self.root / Path("meta.json")),
            'log': str(self.root / Path("log.txt"))
        }

    def meta(self, d):
        """Logs metadata by parsing json files."""

        path = self.paths['meta']

        try:
            metadata = json.load(open(path))
        except FileNotFoundError:
            metadata = dict()

        # Merge the two dictionaries, with the provided one taking priority
        new_d = {**metadata, **d}

        with open(path, 'w') as f:
            json.dump(new_d, f, indent=4, sort_keys=True)

    def loss(self, cc, dt, tl, vl, clr):
        """Logs information about the current epoch.

        Parameters
        ----------
        cc : int
            The current epoch.
        dt : float
            The elapsed time to log.
        tl : float or list
            The training losses.
        vl: : float or list
            The validation losses.
        clr : float
            The current learning rate.
        """

        if isinstance(tl, (list, np.ndarray)):
            tl = "\t".join([f"{xx:.08e}" for xx in tl])
        else:
            tl = f"{tl:.08e}"
        if isinstance(vl, (list, np.ndarray)):
            vl = "\t".join([f"{xx:.08e}" for xx in vl])
        else:
            vl = f"{vl:.08e}"

        with open(self.paths['loss'], 'a') as f:
            f.write(f"{cc:03}\t{dt:.08e}\t{tl}\t{vl}\t{clr}\n")

    def log(self, msg):
        """General log file. Each line is timestamped and a message printed."""

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.paths['log'], 'a') as f:
            f.write(f"[{now}] {msg}\n")
