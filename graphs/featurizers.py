from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import time

from rdkit import Chem

from graphs.data_readers.qm9 import read_qm9_xyz


def _to_mol(qm9_id, dat):
    """Helper function to convert a data point as ingested by read_qm9_xyz to
    an rdkit Mol object.
    
    Parameters
    ----------
    qm9_id : int
    dat : Tuple
        Output from read_qm9_xyz.
    
    Returns
    -------
    int, rdkit.Mol
    """

    smiles = dat[0]  # SMILES
    mol = Chem.MolFromSmiles(smiles)
    return qm9_id, mol


class QM9DataProcessor:

    def __init__(
        self, load_directory, save_directory, debug=0,
        n_workers=cpu_count(), name="qm9data", force=False
    ):
        """Summary
        
        Parameters
        ----------
        load_directory : TYPE
            Description
        save_directory : TYPE
            Description
        debug : int, optional
            Description
        n_workers : TYPE, optional
            Description
        name : str, optional
            Description
        force : bool, optional
            If False, will check the target directory for an existing file
            of the same name. If it exists, it loads this instead, saving the
            time from regenerating the file.
        """

        # If the save directory does not exist, we create it
        self._name = name
        self._load_directory = load_directory
        self._save_directory = Path(save_directory)
        self._save_directory.mkdir(exist_ok=True, parents=True)

        # Get all the xyz files to load
        self._all_xyz = list(Path(self._load_directory).glob("*.xyz"))
        print(f"Preparing to laod {len(self._all_xyz)} xyz files")

        if debug > 0:
            print(f"In debug mode, only {debug} data will be loaded")
            self._all_xyz = self._all_xyz[:debug]

        self._n_workers = n_workers
        self._force = force

    def _process_qm9_step1(self):
        """Reads in the QM9 structural data and returns a dictionary of,
        essentially, the plain text content of the database.
            
        Returns
        -------
        dict
            Dictionary indexed by QM9 ID containing the raw QM9 data.
        """

        name = Path(f"{self._name}_raw.pkl")
        save_path = self._save_directory / name

        if not self._force:
            if save_path.exists():
                print(f"Data in {save_path} exists", end=" ")
                return pickle.load(open(save_path, "rb"))
        
        # Load in the data in parallel
        L = len(self._all_xyz)
        print(f"Loading {L} data using {self._n_workers} workers")
        results = Parallel(n_jobs=self._n_workers)(
            delayed(read_qm9_xyz)(p) for p in self._all_xyz
        )

        # Construct a dictionary indexed by qm9 ID to better keep track of the
        # results
        basic_results_dict = dict()
        for dat in results:

            # Index the results by qm9 ID (dat[0])
            basic_results_dict[dat[0]] = dat[1:]

        pickle.dump(basic_results_dict, open(save_path, "wb"), protocol=4)
        print(f"Saved basic results to {save_path}", end=" ")

        return basic_results_dict

    def _process_qm9_step2(self, basic_results_dict):

        name = Path(f"{self._name}_mol.pkl")
        save_path = self._save_directory / name

        if not self._force:
            if save_path.exists():
                print(f"Data in {save_path} exists", end=" ")
                return pickle.load(open(save_path, "rb"))

        print(f"Getting mol representation")

        results2 = Parallel(n_jobs=self._n_workers)(
            delayed(_to_mol)(key, val)
            for key, val in basic_results_dict.items()
        )

        # Do the same thing for the Mol objects
        mol_results_dict = dict()
        for (qm9_id, mol) in results2:

            # Index the results by qm9 ID again
            mol_results_dict[qm9_id] = mol

        pickle.dump(mol_results_dict, open(save_path, "wb"), protocol=4)
        print(f"Saved rdkit.Mol results to {save_path}", end=" ")

        return mol_results_dict

    def __call__(self):

        t0 = time.time()
        basic_results_dict = self._process_qm9_step1()
        dt = (time.time() - t0) / 60.0
        print(f"{dt:.02f} m")

        t0 = time.time()
        mol_results_dict = self._process_qm9_step2(basic_results_dict)
        dt = (time.time() - t0) / 60.0
        print(f"{dt:.02f} m")


if __name__ == '__main__':
    proc = QM9DataProcessor(
        "/home/mc/LocalData/QM9_Structures",
        "/home/mc/LocalData/QM9_Structures_processed",
        force=True
    )
    proc()
