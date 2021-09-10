import json
from pathlib import Path


class Metadata(dict):

    @classmethod
    def load(cls, path, ok_not_exist=True):
        """Loads in a metadata json object from disk.
        
        Parameters
        ----------
        path : str
            Path to the json file in question.
        ok_not_exist : bool, optional
            If True, then if the file path does not exist, this fact will be
            silently ignored and an "empty" metadata file will be returned.
            Default is True.
        
        Returns
        -------
        Metadata
            The metadata class.
        """

        path = Path(path)

        if not path.exists():
            return cls(dict(), path=path)

        with open(path, "r") as infile:
            d = json.load(infile)
        return cls(d, path=path)

    def save(self):
        """Saves the state of the dictionary to the user-provided path."""

        with open(self._path, "w") as outfile:
            json.dump(dict(self), outfile, indent=4, sort_keys=True)

    def __init__(self, d=dict(), path=None):
        super().__init__(d)
        self._path = path
