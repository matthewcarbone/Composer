import pickle


class QM9Dataset(dict):

    @classmethod
    def load(cls, path):
        """Summary
        
        Parameters
        ----------
        path : TYPE
            Description
        """

        dat = pickle.load(open(path, "rb"))
        return cls(dat)
        
    def save(self, path):
        """Saves the class as a dictionary with e.g. rdkit dependencies, but
        not local class dependencies (i.e. on the QM9Dataset object).
        
        Parameters
        ----------
        path : str
            Path to where the class should be saved.
        """

        pickle.dump(dict(self), open(path, "wb"), protocol=4)

    def __init__(self, dat=dict(), path=None):
        """Initializes the QM9Dataset object
        
        Parameters
        ----------
        dat : dict, optional
            Initializes the dictionary part of the class.
        path : str, optional
            Sets the location from which to load the QM9 data.
        """

        super().__init__(self, dat)
        
        if dict(self) == {} and path is None:
            raise RuntimeError("Specify path or data during instantiation")

        elif dict(self) !
