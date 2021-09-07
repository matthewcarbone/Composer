import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
import pickle

from rdkit import Chem

from graphs.data_readers.qm9 import read_qm9_xyz


def _to_mol(dat):
    """Helper function to convert a data point as ingested by read_qm9_xyz to
    an rdkit Mol object.
    
    Parameters
    ----------
    dat : Tuple
        Output from read_qm9_xyz.
    
    Returns
    -------
    int, rdkit.Mol
    """

    qm9_id = dat[0]
    smiles = dat[1]  # SMILES
    mol = Chem.MolFromSmiles(smiles)
    return qm9_id, mol


def featurize_qm9(
    load_directory, save_directory, debug=1000, n_workers=cpu_count()
):
    """Summary
    
    Parameters
    ----------
    load_directory : str
        Description
    debug : int, optional
        Description
    n_workers : TYPE, optional
        Description
    """

    # If the save directory does not exist, we create it
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    # Get the current YMD
    now = datetime.datetime.now().strftime("%y%m%d")

    # Get all the xyz files to load
    all_xyz = list(Path(load_directory).glob("*.xyz"))
    print(f"Preparing to laod {len(all_xyz)} xyz files to mol and graph")

    if debug > 0:
        print(f"In debug mode, only {debug} data will be loaded")
        all_xyz = all_xyz[:debug]

    # Load in the data in parallel
    print(f"Loading {len(all_xyz)} data using {n_workers} workers")
    results = Parallel(n_jobs=n_workers)(
        delayed(read_qm9_xyz)(p) for p in all_xyz
    )

    # Construct a dictionary indexed by qm9 ID to better keep track of the
    # results
    basic_results_dict = dict()
    for dat in results:

        # Index the results by qm9 ID (dat[0])
        basic_results_dict[dat[0]] = results[1:]

    save_path = save_directory / Path(f"qm9_struct_basic_{now}.pkl")
    pickle.dump(basic_results_dict, open(save_path, "wb"), protocol=4)
    print(f"Saved basic results to {save_path}")

    print(f"Getting mol representation")
    results2 = Parallel(n_jobs=n_workers)(
        delayed(_to_mol)(dat) for dat in results
    )

    # Do the same thing for the Mol objects
    mol_results_dict = dict()
    for (qm9_id, mol) in results2:

        # Index the results by qm9 ID again
        mol_results_dict[qm9_id] = mol

    save_path = save_directory / Path(f"qm9_struct_mol_{now}.pkl")
    pickle.dump(mol_results_dict, open(save_path, "wb"), protocol=4)
    print(f"Saved rdkit.Mol results to {save_path}")


if __name__ == '__main__':
    featurize_qm9(
        "/home/mc/LocalData/QM9_Structures",
        "/home/mc/LocalData/QM9_Structures_processed"
    )
