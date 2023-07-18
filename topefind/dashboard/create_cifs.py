import pandas as pd

from biotite.structure.io.general import save_structure
from joblib import Parallel, delayed
from tqdm import tqdm

from topefind.data_hub import SabdabHub
from topefind.utils import SABDAB_PATH


def re_save_cif_from_pdb(pdb_id):
    loaded_structure = SabdabHub.read_pdb(SABDAB_PATH / "all" / "imgt" / f"{pdb_id}.pdb")
    save_structure(f"{pdb_id}.cif", loaded_structure)


def main():
    pdbs = pd.read_pickle("benchmark.pkl.gz")["pdb"].unique()
    Parallel(n_jobs=4)(delayed(re_save_cif_from_pdb)(pdb) for pdb in tqdm(pdbs))


if __name__ == "__main__":
    main()
