import datetime
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
from rdkit import Chem
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

from composer.graphs.data_readers.qm9 import read_qm9_xyz
from composer.utils.metadata import Metadata


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


class _BaseProcessor:

    def __init__(self, load_directory, save_directory, name, force):

        self._name = name

        # If the save directory does not exist, we create it
        self._load_directory = Path(load_directory)
        self._save_directory = Path(save_directory)
        self._save_directory.mkdir(exist_ok=True, parents=True)
        self._force = force

        metadata_name = f"{self._name}_metadata.json"
        metadata_path = self._save_directory / Path(metadata_name)
        self._metadata = Metadata({
            "date": datetime.datetime.now().strftime("%y%m%d"),
            "loaded": str(self._load_directory),
            "saved": str(self._save_directory)
        }, path=metadata_path)


class QM9DataProcessor(_BaseProcessor):

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

        super().__init__(load_directory, save_directory, name, force)
        
        # Get all the xyz files to load
        self._all_xyz = list(Path(self._load_directory).glob("*.xyz"))
        print(f"Preparing to load {len(self._all_xyz)} xyz files")

        if debug > 0:
            print(f"In debug mode, only {debug} data will be loaded")
            self._all_xyz = self._all_xyz[:debug]

        self._n_workers = n_workers 

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

        self._metadata["n_dat"] = L

        return basic_results_dict

    def _process_qm9_step2(self, basic_results_dict):
        """Processes the basic results from step 1 into rdkit.Mol objects.
        
        Parameters
        ----------
        basic_results_dict : dict
            Dictionary from step 1.
        
        Returns
        -------
        dict
            Dictionary with keys as the qm9_ids and values of the rdkit.Mol
            objects.
        """

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

    def _process_qm9_step3(
        self,
        mol_results_dict,
        node_method="weave",
        edge_method="canonical"
    ):
        """Summary
        
        Parameters
        ----------
        mol_results_dict : dict
            Dictionary from step 2.
        node_method : {'weave'}, optional
            The node featurization method.
        edge_method : {'canonical'}, optional
            The edge featurization method.
        """

        name = Path(f"{self._name}_graph.pkl")
        save_path = self._save_directory / name

        if not self._force:
            if save_path.exists():
                try:
                    c1 = self._metadata["node_method"] == node_method
                    c2 = self._metadata["edge_method"] == edge_method
                except KeyError:
                    c1 = False
                    c2 = False

                if c1 and c2:
                    print(f"Data in {save_path} exists", end=" ")

        self._metadata["node_method"] = node_method
        self._metadata["edge_method"] = edge_method

        if node_method == "weave":
            from dgllife.utils.featurizers import WeaveAtomFeaturizer
            fn = WeaveAtomFeaturizer(
                atom_data_field='features',
                atom_types=['H', 'C', 'N', 'O', 'F']
            )
            self._metadata["n_node_features"] = fn.feat_size()
        else:
            raise ValueError(f"Unknown node method: {node_method}")

        if edge_method == 'canonical':
            from dgllife.utils.featurizers import CanonicalBondFeaturizer
            fe = CanonicalBondFeaturizer(bond_data_field='features')
            self._metadata["n_edge_features"] = fe.feat_size()
        else:
            raise ValueError(f"Unknown edge method: {node_method}")

        # Only import this when its needed
        from dgllife.utils.mol_to_graph import mol_to_bigraph

        def _to_graph(qm9_id, mol):
            return qm9_id, mol_to_bigraph(
                mol, node_featurizer=fn, edge_featurizer=fe
            )

        results3 = Parallel(n_jobs=self._n_workers)(
            delayed(_to_graph)(qm9_id, mol)
            for qm9_id, mol in mol_results_dict.items()
        )

        # Do the same thing for the Mol objects
        graph_results_dict = dict()
        for (qm9_id, graph) in results3:

            # Index the results by qm9 ID again
            graph_results_dict[qm9_id] = graph

        pickle.dump(graph_results_dict, open(save_path, "wb"), protocol=4)
        print(f"Saved dgllife graph results to {save_path}", end=" ")

    def __call__(self):

        t0 = time.time()
        basic_results_dict = self._process_qm9_step1()
        dt = (time.time() - t0) / 60.0
        print(f"{dt:.02f} m")

        t0 = time.time()
        mol_results_dict = self._process_qm9_step2(basic_results_dict)
        dt = (time.time() - t0) / 60.0
        print(f"{dt:.02f} m")

        t0 = time.time()
        self._process_qm9_step3(mol_results_dict)
        dt = (time.time() - t0) / 60.0
        print(f"{dt:.02f} m")

        self._metadata.save()
        print(f"Saved metadata to {self._metadata._path}")


class QM9FEFFDataProcessor(_BaseProcessor):

    def __init__(
        self, load_directory, save_directory, element,
        name="qm9feff", force=False, tail_scaling_proportion=0.25,
        debug=0
    ):
        super().__init__(load_directory, save_directory, name, force)

        # These grids are empirically pre-determined
        if element == "Nitrogen":
            self._grid_params = [397.0, 431.0, 200]
            self._grid = np.linspace(*self._grid_params)
        elif element == "Oxygen":
            self._grid_params = [527.0, 562.0, 200]
            self._grid = np.linspace(*self._grid_params)
        else:
            raise ValueError(f"Unknown element: {element}")

        self._tail_scaling_number \
            = int(len(self._grid) * tail_scaling_proportion)

        self._molecules = self._get_molecules()
        L = len(self._molecules)
        print(f"Found {L} possible spectra (molecules)")
        if debug > 0:
            print(f"Debug mode, only {debug} molecules will be considered")
            self._molecules = self._molecules[:debug]

        self._metadata["grid_params"] = self._grid_params
        self._metadata["tail_scaling_number"] = self._tail_scaling_number

    def _get_molecules(self):
        return [xx for xx in self._load_directory.iterdir() if xx.is_dir()]

    def _get_averaged_spectrum(self, d):
        """Gets the molecule FEFF spectrum by averaging the contributions from
        each site.
        
        Parameters
        ----------
        d : pathlib.PosixPath or str
            The path to the molecule. Ends with a number like 112463.
        
        Returns
        -------
        np.ndarray, int
            The average spectrum scaled such that the average value of the
            last quarter of the spectrum is equal to 1.
        """

        sites = [xx for xx in d.iterdir() if xx.is_dir()]

        data = []
        for site in sites:

            # For each site, load the spectrum. If the file does not exist,
            # that entire molecule is discarded. Note sometimes the xmu file
            # is empty, likely because the FEFF calculation failed. That's
            # why we catch the warnings. This error is caught in the next
            # try/except statement.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dat = np.loadtxt(site / Path("xmu.dat"))
            except OSError as err:
                return None, str(err)

            try:
                grid = dat[:, 0]
                spec = dat[:, 3]
            except IndexError as err:
                return None, str(err)

            assert len(dat.shape) == 2

            # Then we interpolate onto the standardized grid
            spline = InterpolatedUnivariateSpline(
                grid, spec, ext='zeros', k=3
            )

            # Each site has multiplicity 1 (since these are low symmetry
            # molecules not crystals)
            spec2 = spline(self._grid)

            # Scale to the tail region as best as we can
            mean = np.mean(spec2[:-self._tail_scaling_number])
            spec2 = spec2 / mean

            data.append(spec2)

        return np.array(data).mean(axis=0).squeeze(), len(sites)

    def __call__(self):
        """Finds all the spectra contained in the user-provided root, loads
        and processes them, then saves the processed spectra + metadata to
        disk."""

        name = Path(f"{self._name}.pkl")
        save_path = self._save_directory / name

        t0 = time.time()
        if not self._force:
            if save_path.exists():
                print(f"Data in {save_path} exists")
                return

        results = dict()
        errors = dict()
        contributing_sites = dict()
        for molecule in tqdm(self._molecules):
            spec, nsites_or_error = self._get_averaged_spectrum(molecule)
            molecule_index = int(molecule.stem)
            if spec is not None:
                results[molecule_index] = spec
                contributing_sites[molecule_index] = nsites_or_error
            else:
                errors[molecule_index] = nsites_or_error

        pickle.dump(results, open(save_path, "wb"), protocol=4)
        self._metadata["errors"] = errors
        self._metadata["contributing_sites"] = contributing_sites
        self._metadata.save()

        dt = (time.time() - t0) / 60.0
        L_err = len(errors)
        print(f"Complete with {L_err} errors {dt:.02f} m")


class QM9FeatureTargetPairing:


if __name__ == '__main__':
    # proc = QM9DataProcessor(
    #     "/home/mc/LocalData/QM9_Structures",
    #     "/home/mc/LocalData/QM9_Structures_processed",
    #     force=False
    # )
    element = "Oxygen"
    proc = QM9FEFFDataProcessor(
       f"/home/mc/LocalData/QM9_Spectra/feff{element}",
       "/home/mc/LocalData/QM9_Spectra_processed",
       element=element,
       name=f"qm9feff_{element}",
       debug=0,
       force=False
    )
    proc()
