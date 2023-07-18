import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray

from topefind.utils import SABDAB_PATH, VALID_IMGT_IDS, download_url


class SabdabHub:
    """
    This class handles the SAbDAb dataset.
    It achieves this by scanning the PDBs and retrieving relevant labels for protein-protein interaction tasks.
    The labels are e.g. interface contact maps, paratope labels and epitope labels.
    The dataset is downloaded from the original: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab.

    Several terminologies are used, here is a subjective definition of the crucial ones:

    `distance map`:
        Representation of distance between two entities given their pair-wise distances between subcomponents.
    `contact map`:
        Same as distance map but binarized given a distance threshold.
    `
    When applied to protein sequences one can differentiate between within protein contact maps which we will refer to
    simply as contact maps and between proteins contact maps which we will refer to as interface contact maps.

    `contact map`:
        Binarized distance map of pair-wise residues of a protein.
        These contact maps are N x N because each residue is considered against all the others in the same sequence.
    `interface contact map`:
        Binarized distance map of pair-wise residues of two proteins.
        These contact maps can be N x M since they represent contacts between two proteins.

    It is still unclear, however, if this representation, with contact maps, is the most convenient.

    The parsed and labeled dataset is saved as a DataFrame in a parquet file.
    The aim is not performance but rather a convenient way to retrieve relevant labels and sequences.
    """

    summary_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
    pdbs_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/"

    def __init__(
            self,
            summary_file_path: Path | str = SABDAB_PATH / "all.tsv",
            sabdab_pdb_path: Path | str = SABDAB_PATH / "all",
            save_path: Path | str = SABDAB_PATH / "sabdab.parquet",
            subset_dim: int = None,
            contact_threshold: float = 4.5,
            invalid_chains: set[str] = ("", "NA"),
            fix_imgt: bool = True,
            re_download: bool = False,
            numbering_scheme: Literal["imgt", "raw", "chothia"] = "imgt",
            n_jobs: int = 1,
    ):
        """
        Initializes the needed PDB files and DataFrame for the SAbDab dataset.
        SAbDab provides th

        Parameters
        ----------
        summary_file_path: Path to the summary file.
        sabdab_pdb_path: Path to the stored PDBs.
        save_path: Path to the final parsed DataFrame.
        subset_dim: How many elements to consider from the original dataset.
        contact_threshold: Distant threshold for contacts.
        invalid_chains: Set of strings known to be invalid chains in the PDB.
        fix_imgt: Whether to re-filter the dataset based on antibodies that fall into the valid IMGT defined numbering.
        re_download: Re-download the dataset.
        numbering_scheme: Accepted numbering schemes from SAbDab.
        n_jobs: Number of jobs for Joblib.

        """
        self.sabdab_pdb_path = Path(sabdab_pdb_path) / numbering_scheme
        self.save_path = Path(save_path)
        self.contact_threshold = contact_threshold
        self.invalid_chains = invalid_chains
        self.numbering_scheme = numbering_scheme
        self.fix_imgt = fix_imgt
        self.n_jobs = n_jobs

        if not Path(self.sabdab_pdb_path).exists() and re_download:
            SabdabHub.download_original(summary_file_path, numbering_scheme)

        self.df = pl.read_csv(
            summary_file_path,
            sep="\t",
            n_rows=subset_dim,
            null_values=["NOT", "8.9, 8.9", "3.9, 3.9"],  # Invalid cells, check better the DF.
        )

        # Setup and cleaning
        self.df = self.df.with_columns([
            pl.col("resolution").cast(pl.Float64, strict=False),
            pl.col("temperature").cast(pl.Float64, strict=False),
            pl.col("affinity").cast(pl.Float64, strict=False),
            pl.col("delta_g").cast(pl.Float64, strict=False),
            pl.col("date").str.strptime(pl.Date, fmt="%m/%d/%y"),
            pl.col("Hchain").cast(pl.Utf8),
            pl.col("Lchain").cast(pl.Utf8),
            pl.col("pdb").cast(pl.Utf8),
            pl.col("scfv").cast(pl.Boolean),
            pl.col("antigen_chain").cast(pl.Utf8),
        ])

        # Remove antibodies without antigen
        self.df = self.df.filter(~pl.col("antigen_chain").is_in(invalid_chains))
        self.df = self.df.drop_nulls("antigen_chain")

        # Split antigen chains by "|".
        self.df = self.df.with_columns(
            pl.col("antigen_chain").str.replace_all(" ", "").str.split("|")
        )
        self.df = self.df.with_columns(
            pl.col("antigen_chain").arr.lengths().alias("num_antigen_chains").cast(pl.UInt8)
        )
        # Constrain the column to contain lists of strings.
        self.df = self.df.with_columns(
            pl.struct(["num_antigen_chains", "antigen_chain"]).apply(
                lambda x: x["antigen_chain"] if x["num_antigen_chains"] > 0 else [""]
            ).alias("antigen_chain")
        )
        # Now let's make the original summary file tidy.
        # First let's add an antibody chain type.
        df_h = self.df.select(pl.exclude("Lchain"))
        df_h = df_h.with_columns(pl.Series(name="chain_type", values=["heavy"] * len(df_h), dtype=pl.Utf8))
        df_h = df_h.rename({"Hchain": "antibody_chain"})

        df_l = self.df.select(pl.exclude("Hchain"))
        df_l = df_l.with_columns(pl.Series(name="chain_type", values=["light"] * len(df_l), dtype=pl.Utf8))
        df_l = df_l.rename({"Lchain": "antibody_chain"})

        self.df = pl.concat([df_h, df_l])

        # Cleaning up.
        self.df = self.df.drop_nulls("antibody_chain")
        self.df = self.df.filter(~pl.col("antibody_chain").is_in(invalid_chains))

        # Let's explode the antigen chains, useful since we will have different contact maps.
        self.df = self.df.explode("antigen_chain")

        # Some entries have problems, the antibody chain might be labelled with the same letter as the antigen.
        self.df = self.df.filter(pl.col("antibody_chain") != pl.col("antigen_chain"))
        self.df = self.df.sort("pdb")

        # Now we have a tidy summary DataFrame to work with.
        # The columns ["pdb", "chain", "antigen_chain"] together provide a primary key.

    @staticmethod
    def download_original(summary_file_path, scheme):
        print("Downloading SAbDab summary file")
        download_url(SabdabHub.summary_url, summary_file_path)

        print("Downloading SAbDab archive file")
        download_url(SabdabHub.pdbs_url, SABDAB_PATH / "all.zip", chunk_size=1024)

        print("Unzipping PDB files")
        with zipfile.ZipFile(SABDAB_PATH / "all.zip", 'r') as z:
            # There are 3 subdirectories in the zip file: raw, chothia and imgt.
            # Select the ones you need with `scheme`.
            for file_name in tqdm(z.namelist()):
                if scheme in file_name:
                    z.extract(file_name, SABDAB_PATH)
        (SABDAB_PATH / "all_structures").rename(SABDAB_PATH / "all")

    @staticmethod
    def read_pdb(pdb_path: Path | str) -> AtomArray:
        structure = PDBFile.read(str(pdb_path))
        return structure.get_structure(1)

    @staticmethod
    def compute_chain_res_to_seq(
            atoms: AtomArray,
    ) -> dict[str, str]:
        # Order is maintained as in the biotite.structure.AtomArray according to the PDB scheme.
        # This implementation takes in account the fact that the AtomArray passed is of one chain only.
        # Special attention given to insertion codes.
        res_ids_w_ins = SabdabHub.get_atom_res_ins(atoms)
        res_names = map(ProteinSequence.convert_letter_3to1, atoms.res_name)
        res_to_seq = dict(zip(res_ids_w_ins, res_names))
        return res_to_seq

    @staticmethod
    def compute_chain_seq(res_to_seq: dict[str, str]) -> str:
        # In this implementation res_to_seq has to be ONE chain only.
        return "".join(list(res_to_seq.values()))

    @staticmethod
    def select_non_h_atoms(atoms: AtomArray):
        return atoms[atoms.element != "H"]

    @staticmethod
    def select_non_hetero_atoms(atoms: AtomArray):
        return atoms[~atoms.hetero]

    @staticmethod
    def select_chain_in_atoms(
            atoms: AtomArray,
            chain_id: str
    ):
        return atoms[atoms.chain_id == chain_id]

    @staticmethod
    def get_atom_res_ins(atoms: AtomArray) -> np.ndarray:
        # What we want to do is threat residue ids and insertion codes together as a string for each atom.
        # Something like this:
        # return {i: f"{a.res_id}{a.ins_code}" for i, a in enumerate(atoms)}

        # Let's do it more efficiently:
        # 1) First, ensure that we have numpy arrays of strings.
        # 2) Then we can add in a vectorized manner.
        return np.char.add(
            np.array(atoms.res_id, dtype=str),
            np.array(atoms.ins_code, dtype=str)
        )

    @staticmethod
    def get_sequence(
            atoms: AtomArray,
            chain_id: str,
    ) -> str:
        chain_atoms = SabdabHub.select_chain_in_atoms(atoms, chain_id)
        res_to_seq = SabdabHub.compute_chain_res_to_seq(chain_atoms)
        sequence = SabdabHub.compute_chain_seq(res_to_seq)
        return sequence

    @staticmethod
    def get_interface_contact_map(
            atoms: AtomArray,
            chain_id_1: str,
            chain_id_2: str,
            contact_threshold: float,
    ) -> np.ndarray:
        # Get the atoms w.r.t the chains.
        atoms_1 = SabdabHub.select_chain_in_atoms(atoms, chain_id_1)
        atoms_2 = SabdabHub.select_chain_in_atoms(atoms, chain_id_2)

        # Get res_id lists to be able to remap using the index.
        # This is necessary since res_id contain strings because of the ins_code addition.
        res_to_seq_1 = SabdabHub.compute_chain_res_to_seq(atoms_1)
        res_to_seq_2 = SabdabHub.compute_chain_res_to_seq(atoms_2)
        res_ids_1 = list(res_to_seq_1.keys())
        res_ids_2 = list(res_to_seq_2.keys())

        # Getting the array that contains for residues and insertion codes.
        atom_res_ins_1 = SabdabHub.get_atom_res_ins(atoms_1)
        atom_res_ins_2 = SabdabHub.get_atom_res_ins(atoms_2)

        # Getting the distance contact map atom wise.
        pair_dists = cdist(atoms_1.coord, atoms_2.coord)

        # Getting binarized contact map atom wise.
        atom_contact_map = pair_dists <= contact_threshold

        # Finding two arrays of paired atoms in contact.
        interface_atoms_ids_1 = np.nonzero(atom_contact_map)[0]
        interface_atoms_ids_2 = np.nonzero(atom_contact_map)[1]

        # To get the atoms of the interface that are in contact:
        # interface_atoms_1 = atoms_1[interface_atoms_ids_1]
        # interface_atoms_2 = atoms_2[interface_atoms_ids_2]

        chain_1_len = len(res_ids_1)
        chain_2_len = len(res_ids_2)

        # Translating to binarized contact map RESIDUE WISE by the following lookups:
        # atom_id -> res_id -> res_index_identifier
        # This kind of double lookup is important since antibodies have insertion codes.
        # Feel free to open an issue to propose better alternatives to deal with this.
        residue_interface_contact_map = np.zeros((chain_1_len, chain_2_len), dtype=bool)
        for a1, a2 in zip(interface_atoms_ids_1, interface_atoms_ids_2):
            residue_interface_contact_map[
                res_ids_1.index(atom_res_ins_1[a1]),
                res_ids_2.index(atom_res_ins_2[a2])
            ] = True

        # We have computed two different adjacency matrices:
        # `residues_contact_map` and `atom_contact_map`.
        # Given the sparsity, we would wish to return this in a convenient format.
        # However, for the length at hand for antibodies (ca. 110) and antigens (ca. 50 - 1000)
        # we can simply store/return/reuse efficiently the binary matrix which occupies `size * 4` bytes.
        return residue_interface_contact_map

    @staticmethod
    def get_interface_distance_map(
            atoms: AtomArray,
            chain_id_1: str,
            chain_id_2: str,
            contact_threshold: float,
    ) -> np.ndarray:
        # FIXME: This function is not complete and needs to be fully revised.
        # FIXME: In particular, the choice of the point of contact has to be settled.
        # FIXME: For example, one could choose the `Cβ` or the `center of mass` as the point of contact for each AA.
        # Choosing the `Cβ` might be valid only in the context of a certain definition of contact.
        # At the same time choosing the `center of mass` would invalidate the 4.5-based classical definition.
        # So it is a bit tricky.
        # Probably, the idea for the simpler way in the 4.5-based definition, would be to consider
        # always the shortest distance between two AA. This could be computationally inefficient if naively
        # implemented, but would keep consistency with the definition of contact for the 4.5-based case.

        # Get the atoms w.r.t the chains.
        atoms_1 = SabdabHub.select_chain_in_atoms(atoms, chain_id_1)
        atoms_2 = SabdabHub.select_chain_in_atoms(atoms, chain_id_2)

        # Get res_id lists to be able to remap using the index.
        # This is necessary since res_id contain strings because of the ins_code addition.
        res_to_seq_1 = SabdabHub.compute_chain_res_to_seq(atoms_1)
        res_to_seq_2 = SabdabHub.compute_chain_res_to_seq(atoms_2)
        res_ids_1 = list(res_to_seq_1.keys())
        res_ids_2 = list(res_to_seq_2.keys())

        # Getting the array that contains for residues and insertion codes.
        atom_res_ins_1 = SabdabHub.get_atom_res_ins(atoms_1)
        atom_res_ins_2 = SabdabHub.get_atom_res_ins(atoms_2)

        # Getting the distance contact map atom wise.
        pair_dists = cdist(atoms_1.coord, atoms_2.coord)

        # Getting binarized contact map atom wise.
        atom_contact_map = pair_dists <= contact_threshold

        # Finding two arrays of paired atoms in contact.
        interface_atoms_ids_1 = np.nonzero(atom_contact_map)[0]
        interface_atoms_ids_2 = np.nonzero(atom_contact_map)[1]

        chain_1_len = len(res_ids_1)
        chain_2_len = len(res_ids_2)

        # Now we need the actual distance now.
        residue_interface_distance_map = np.zeros((chain_1_len, chain_2_len))
        for a1, a2 in zip(interface_atoms_ids_1, interface_atoms_ids_2):
            dist = np.linalg.norm(atoms_1[a1].coord - atoms_2[a2].coord, ord=2)
            residue_interface_distance_map[
                res_ids_1.index(atom_res_ins_1[a1]),
                res_ids_2.index(atom_res_ins_2[a2])
            ] = dist

        return residue_interface_distance_map

    @staticmethod
    def get_tope_labels(
            residue_contact_map: np.ndarray,
            axis: int
    ) -> np.ndarray:
        return np.any(residue_contact_map, axis)

    @staticmethod
    def label_dimer(
            pdb_path: Path | str,
            chain_1: str,
            chain_2: str,
            contact_threshold: float,
            invalid_chains: set[str],
    ) -> pl.DataFrame:

        try:
            # Loading and cleaning atoms
            atoms = SabdabHub.read_pdb(pdb_path)
            atoms = SabdabHub.select_non_hetero_atoms(atoms)
            atoms = SabdabHub.select_non_h_atoms(atoms)

            ab_atoms = SabdabHub.select_chain_in_atoms(atoms, chain_1)
            numbering = list(SabdabHub.compute_chain_res_to_seq(ab_atoms).keys())

            # Get the sequences
            chain_1_seq = SabdabHub.get_sequence(atoms, chain_1) if chain_1 not in invalid_chains else ""
            chain_2_seq = SabdabHub.get_sequence(atoms, chain_2) if chain_2 not in invalid_chains else ""

            # Compute the interface contact map
            if chain_1 in invalid_chains or chain_2 in invalid_chains:
                residue_interface_contact_map = \
                    residue_interface_distance_map = \
                    interface_1_labels = \
                    interface_2_labels = np.zeros(0, dtype=bool)
            else:
                residue_interface_contact_map = \
                    SabdabHub.get_interface_contact_map(atoms, chain_1, chain_2, contact_threshold)
                residue_interface_distance_map = \
                    SabdabHub.get_interface_distance_map(atoms, chain_1, chain_2, contact_threshold).ravel()
                interface_1_labels = SabdabHub.get_tope_labels(residue_interface_contact_map, axis=1)
                interface_2_labels = SabdabHub.get_tope_labels(residue_interface_contact_map, axis=0)
                residue_interface_contact_map = residue_interface_contact_map.ravel()

        except (ValueError, KeyError, FileNotFoundError) as e:
            # Sometimes there are some minor inconsistency in the PDBs.
            # TODO: better to use logging to log these and further inspect the "ugly" PDBs.
            print(f"{Path(pdb_path).name} is broken or did not parse correctly.\n"
                  f"Error details: {e}")
            chain_1_seq = chain_2_seq = numbering = ""
            residue_interface_contact_map = \
                residue_interface_distance_map = \
                interface_1_labels = \
                interface_2_labels = np.zeros(0, dtype=bool)

        return pl.DataFrame([
            pl.Series(name="antibody_sequence", values=[chain_1_seq], dtype=pl.Utf8),
            pl.Series(name="antibody_imgt", values=[numbering], dtype=pl.List(pl.Utf8)),
            pl.Series(name="antigen_sequence", values=[chain_2_seq], dtype=pl.Utf8),
            pl.Series(name="paratope_labels", values=[interface_1_labels], dtype=pl.List(pl.Boolean)),
            pl.Series(name="epitope_labels", values=[interface_2_labels], dtype=pl.List(pl.Boolean)),
            pl.Series(
                name="residue_interface_contact_map",
                values=[residue_interface_contact_map],
                dtype=pl.List(pl.Boolean)),
            pl.Series(
                name="residue_interface_distance_map",
                values=[residue_interface_distance_map],
                dtype=pl.List(pl.Float32)),
        ])

    def __call__(self) -> pd.DataFrame:
        # Getting the interested flattened arrays from original DataFrame
        pdbs = self.df.select("pdb").to_numpy().flatten()
        ab_chains = self.df.select("antibody_chain").to_numpy().flatten()
        ag_chains = self.df.select("antigen_chain").to_numpy().flatten()
        pdb_paths = [self.sabdab_pdb_path / f"{pdb}.pdb" for pdb in pdbs]

        # Joblibbing the labeling of each dimer.
        # This is definitely not the best way to do it!
        # Please reconsider more efficient ways in the future to no re-read the files.
        zipped_args = zip(pdb_paths, ab_chains, ag_chains)
        results_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(SabdabHub.label_dimer)(pdb_path, ab_chain, ag_chain, self.contact_threshold, self.invalid_chains)
            for pdb_path, ab_chain, ag_chain in tqdm(zipped_args, total=len(pdb_paths))
        )

        print("Joining DataFrames")
        results = pl.concat(results_dfs)
        self.df = pl.concat([self.df, results], how="horizontal")

        # Adding full paratopes
        # Some antibodies can be bound to several antigens.
        # The full paratope is the logical OR operation on all the interested paratopes in this case.
        def get_full_paratope(group: pl.Series):
            group_elements = [el for el in group.to_list() if len(el) > 0]
            return list(np.any(np.vstack(group_elements), axis=0).ravel()) if len(group_elements) > 0 else []

        print("Calculating full paratopes")
        full_paratopes = self.df \
            .groupby(["pdb", "antibody_chain"], maintain_order=True) \
            .agg(pl.col("paratope_labels")
                 .apply(lambda group: get_full_paratope(group))
                 .alias("full_paratope_labels"))

        # Explicitly setting each value to a python bool, probably casting bug in polars?
        full_paratopes = full_paratopes.with_columns([
            pl.col("full_paratope_labels").apply(lambda labels: [True if lab == 1 else False for lab in labels]),
            pl.col("full_paratope_labels").apply(lambda x: x.sum()).alias("num_full_para_residues").cast(pl.Int32),
        ])
        self.df = self.df.join(full_paratopes, on=["pdb", "antibody_chain"])

        # Adding some extra columns for the lengths of our labels
        self.df = self.df.with_columns([
            pl.col("paratope_labels").apply(lambda x: x.sum()).alias("num_paratope_residues").cast(pl.Int32),
            pl.col("epitope_labels").apply(lambda x: x.sum()).alias("num_epitope_residues").cast(pl.Int32),
            pl.col("residue_interface_contact_map").apply(lambda x: x.sum()).alias("num_interface_residues").cast(
                pl.Int32),
        ])

        print(f"Estimated DF size in memory: {self.df.estimated_size('mb'):.2F} MBs")
        print(self.df.head())
        _ = [print(k, ":", v) for k, v in self.df.schema.items()]

        if self.fix_imgt:
            self.df = pl.DataFrame(SabdabHub.fix_imgts(self.df.to_pandas()))

        print("Saving")
        self.df.write_parquet(self.save_path)

        return self.df.to_pandas()

    @staticmethod
    def fix_imgts(df: pd.DataFrame) -> pd.DataFrame:
        # Re-filter the dataset based on antibodies that fall into the valid IMGT defined numbering.
        # First reduce the length, sometimes antibodies in SAbDab have long FWR4.
        # Then filter the ones that contain invalid IMGT annotations for this use case.
        # TODO: switch this function to use polars
        print("Filtering antibodies with invalid IMGT positions and clipping to a maximum")
        max_imgt_id = VALID_IMGT_IDS[-1]
        new_imgts = []
        for ab_imgt in tqdm(df["antibody_imgt"].values):
            if np.any(np.isin(ab_imgt, str(int(max_imgt_id) + 1))):
                ab_imgt = ab_imgt[:list(ab_imgt).index(max_imgt_id)]
            if not np.all(np.isin(ab_imgt, VALID_IMGT_IDS)):
                ab_imgt = pd.NA
            new_imgts.append(ab_imgt)
        df["antibody_imgt"] = new_imgts
        df = df[df["antibody_imgt"].notna()]

        # Fix now the length up to the cut region, i.e. max_imgt_id
        df["antibody_sequence"] = [
            row[0][:len(row[1])]
            for row in df[["antibody_sequence", "antibody_imgt"]].values
        ]
        df["paratope_labels"] = [
            row[0][:len(row[1])]
            for row in df[["paratope_labels", "antibody_imgt"]].values
        ]
        df["full_paratope_labels"] = [
            row[0][:len(row[1])] for row in
            df[["full_paratope_labels", "antibody_imgt"]].values
        ]
        return df


if __name__ == "__main__":
    SabdabHub(
        subset_dim=None,
        re_download=False,
        n_jobs=1,
    )()
