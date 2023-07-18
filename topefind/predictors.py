"""
This implementation focuses on gathering known antibody paratope predictor methods.
The Facade pattern is preferred to expose the methods.
The responsibility of creating the subsystems falls under each concrete Predictor.

In this module, the main focus is not the performance of forward pass but rather the ease of usage
and the reproducibility of the original method's predictions.
The second main focus is the provisioning of a similar interface between end-to-end models.
"""
import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod

if sys.version_info >= (3, 11):
    from enum import StrEnum, auto
else:
    from backports.strenum import StrEnum
    from backports import auto

import torch
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from topefind.utils import TOPEFIND_PATH, get_device
from topefind.vendored.parapred_pytorch.parapred import model as parapred_model
from topefind.vendored.parapred_pytorch.parapred import preprocessing as parapred_preproc
from topefind.vendored.Paragraph.Paragraph.model import EGNN_Model as EGNN_Model_paragraph
from topefind.vendored.Paragraph.Paragraph.predict import get_dataloader, evaluate_model
from topefind.data_hub import SabdabHub
from topefind import utils, embedders

PARAPRED_PATH = utils.TOPEFIND_PATH / "vendored/parapred_pytorch/parapred/weights/parapred_pytorch.h5"
PARAGRAPH_PATH = utils.TOPEFIND_PATH / "vendored/Paragraph/Paragraph/trained_model/pretrained_weights.pt"
PARAGRAPH_H_PATH = utils.TOPEFIND_PATH / "vendored/Paragraph/Paragraph/trained_model/pretrained_weights_heavy.pt"
PARAGRAPH_L_PATH = utils.TOPEFIND_PATH / "vendored/Paragraph/Paragraph/trained_model/pretrained_weights_light.pt"

# To use `torch.use_deterministic_algorithms` on CUDA >= 10.2 you need to set CUBLAS_WORKSPACE_CONFIG=:4096:8
# This happens because several utilities are running on cuBLAS!
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class EndToEndPredictorName(StrEnum):
    parapred = auto()
    paragraph_untrained = auto()
    paragraph_unpaired = auto()
    paragraph_paired = auto()
    pppred = auto()
    af2_multimer = auto()
    esm_fold_multimer = auto()
    seq_to_cdr = auto()


class Predictor(ABC):
    name: str

    @abstractmethod
    def predict(self, an_input):
        ...

    @abstractmethod
    def predict_multiple(self, some_inputs):
        ...

    @abstractmethod
    def prepare_dataset(self, dataset):
        ...


class Parapred(Predictor):
    def __init__(
            self,
            name: EndToEndPredictorName = EndToEndPredictorName.parapred,
            max_cdr_length: int = 40,
            device: str = "cpu"
    ):
        """
        Model initialization and configuration.

        Parameters
        ----------
        name: Name of the predictor.
        max_cdr_length : Maximum accepted CDR length, to match the original value this is set to 40.
        """
        self.name = name
        self.max_cdr_length = max_cdr_length
        self.device = get_device(device)
        self._model = parapred_model.Parapred()

        state_dict = torch.load(PARAPRED_PATH, map_location=torch.device(device))
        self._model.load_state_dict(state_dict)
        self._model.eval()

    @torch.no_grad()
    def predict(self, sequence: str) -> np.ndarray:
        """
        Predicts the paratope of an antibody.
        Following the Parapred implementation, the inputs to the model should be sequences of regions.
        To achieve this, the sequence is firstly numbered with ANARCI.
        Then each region is predicted with the `parapred_pytorch` model.

        Parameters
        ----------
        sequence: The full amino acid sequence of an antibody (Fv region)

        Returns
        -------
        paratope_probs: Paratope probabilities for each amino acid.

        """
        cdrs, fmks = utils.get_antibody_regions(sequence)
        paratope_probs = np.zeros(0)

        for fmk, cdr in zip(fmks[:-1], cdrs):
            max_len = self.max_cdr_length if len(cdr) > self.max_cdr_length else len(cdr)
            encoded, lengths = parapred_preproc.encode_batch([cdr], max_length=max_len)
            no_mask = torch.ones_like(encoded)
            probs = self._model(encoded, no_mask, lengths)

            # Cleaning up probs from padded
            probs = [parapred_model.clean_output(pr, lengths[i]).numpy() for i, pr in enumerate(probs)]
            paratope_probs = np.concatenate([paratope_probs, np.zeros(len(fmk)), probs[0]])

        # Matching initial sequence length
        paratope_probs = np.concatenate([paratope_probs, np.zeros(len(fmks[-1]))])
        return paratope_probs

    def predict_multiple(self, sequences: list[str], verbose: bool = False, n_jobs: int = 1) -> list[np.ndarray]:
        """
        Dummy parallelization of single predictions.

        Parameters
        ----------
        sequences: list of strings that contain antibody sequences.
        verbose: If True shows tqdm progress bar.
        n_jobs: Number of jobs for joblib.

        Returns
        -------
        list of predictions

        """
        return Parallel(n_jobs)(delayed(self.predict)(seq) for seq in tqdm(sequences, disable=(not verbose)))

    def prepare_dataset(
            self,
            dataset: pl.DataFrame,
    ) -> tuple[list[str], list[bool]]:
        """
        Prepares a given DataFrame to be consumed.

        Parameters
        ----------
        dataset: A DataFrame that has to contain the following columns: "antibody_sequence" and "full_paratope_labels"

        Returns
        -------
        inputs, labels: These are grouped by protein i.e. each element in the inputs is an antibody, and each element
                        in labels is list of labels for each amino acid of the antibody sequence in the inputs.
        """
        inputs = dataset.get_column("antibody_sequence").to_list()
        labels = dataset.get_column("full_paratope_labels").to_list()
        return inputs, labels


class Paragraph(Predictor):
    def __init__(
            self,
            name: EndToEndPredictorName,
            pdbs_path: Path | str,
            device: str = "cpu",
    ):
        """
        Model initialization and configuration.
        To match the usage of Paragraph CSV paths and PDBs paths are included.

        Parameters
        ----------
        name: Name of the predictor.
        pdbs_path : Path to the directory containing all the pdbs.
        """
        self.name = name
        self.pdbs_path = Path(pdbs_path)
        self.device = get_device(device)

        if name == EndToEndPredictorName.paragraph_paired:
            # Getting settings for original model used by authors and initializing model
            # 20D for of AA type and 2D for chain ID
            self._model = EGNN_Model_paragraph(
                num_feats=22,
                graph_hidden_layer_output_dims=[22] * 6,
                linear_hidden_layer_output_dims=[10] * 2,
            )
            state_dict = torch.load(PARAGRAPH_PATH, map_location=torch.device(self.device))
            self._model.load_state_dict(state_dict)

        else:
            self._model_h = EGNN_Model_paragraph(
                num_feats=22,
                graph_hidden_layer_output_dims=[22] * 6,
                linear_hidden_layer_output_dims=[10] * 2,
            )
            self._model_l = EGNN_Model_paragraph(
                num_feats=22,
                graph_hidden_layer_output_dims=[22] * 6,
                linear_hidden_layer_output_dims=[10] * 2,
            )
            if name != EndToEndPredictorName.paragraph_untrained:
                state_dict = torch.load(PARAGRAPH_H_PATH, map_location=torch.device(self.device))
                self._model_h.load_state_dict(state_dict)
                state_dict = torch.load(PARAGRAPH_L_PATH, map_location=torch.device(self.device))
                self._model_l.load_state_dict(state_dict)

    def predict(self, pdb_chains: dict) -> np.ndarray:
        """
        Predicts the paratope of an antibody.
        This implementation follows the authors' way of predicting and calls functions from vendored packages.
        More specifically, this includes the creation of a torch.utils.data.DataLoader and calls the originals
        `get_dataloader()` and the `evaluate_model()`.
        Some minor modifications are included in `topefind.vendored.Paragraph` to allow DataFrames as input.

        Parameters
        ----------
        pdb_chains: dict that contains {pdb_code, H_id, L_id, focus}

        Returns
        -------
        predictions
        """
        h_id = pdb_chains["H_id"][0]
        l_id = pdb_chains["L_id"][0]
        pdb_code = pdb_chains["pdb_code"][0]
        chain_id = pdb_chains["focus"]

        if h_id == "" and l_id == "":
            raise ValueError("Wrong input, specify one chain not zero.")

        # Loading and cleaning atoms.
        atoms = SabdabHub.read_pdb(self.pdbs_path / Path(pdb_code + ".pdb"))
        atoms = SabdabHub.select_non_hetero_atoms(atoms)
        atoms = SabdabHub.select_non_h_atoms(atoms)
        atoms = SabdabHub.select_chain_in_atoms(atoms, chain_id)
        res_ids_w_ins = SabdabHub.get_atom_res_ins(atoms)
        _, idx = np.unique(res_ids_w_ins, return_index=True)
        res_ids_w_ins = res_ids_w_ins[np.sort(idx)]

        # Initialize output probs to zeros.
        paratope_probs = np.zeros(len(res_ids_w_ins))

        # Pass now everything through the original Paragraph.
        df_input = pd.DataFrame(pdb_chains)
        dl = get_dataloader(df_input, self.pdbs_path)

        if self.name == EndToEndPredictorName.paragraph_paired:
            df_out = evaluate_model(self._model, dl, self.device)
        else:
            if l_id == "":
                df_out = evaluate_model(self._model_h, dl, self.device)
            else:
                df_out = evaluate_model(self._model_l, dl, self.device)

        df_out = df_out[df_out["chain_id"] == chain_id]
        probs_out = df_out["pred"].to_numpy()

        # We need to fill in only the CDR regions predicted.
        paragraph_used_res_ids = df_out["IMGT"].to_numpy().flatten()
        mask = np.isin(res_ids_w_ins, paragraph_used_res_ids)
        paratope_probs[mask] = probs_out
        return paratope_probs

    def predict_multiple(self, pdb_chains: list[dict], verbose: bool = False, n_jobs: int = 1) -> list[np.ndarray]:
        """
        Dummy parallelization of single predictions.

        Parameters
        ----------
        pdb_chains: list of dicts that contains pdb name, heavy chain id, light chain id and focus.
        verbose: If True shows tqdm progress bar.
        n_jobs: Number of jobs for joblib.

        Returns
        -------
        list of predictions

        """
        return Parallel(n_jobs)(delayed(self.predict)(d) for d in tqdm(pdb_chains, disable=(not verbose)))

    def prepare_dataset(
            self,
            dataset: pl.DataFrame,
    ) -> tuple[list[dict], list[bool]]:
        """
        Prepares a given DataFrame to be consumed.

        Parameters
        ----------
        dataset: A DataFrame that has to contain the following columns:
                 "antibody_sequence",
                 "full_paratope_labels",
                 "chain_type",
                 "pdb",

        Returns
        -------
        inputs, labels: These are grouped by protein i.e. each element in the inputs is an antibody, and each element
                        in labels is list of labels for each amino acid of the antibody sequence in the inputs.
        """
        inputs = []
        pdbs = dataset.get_column("pdb").to_list()
        ab_chains = dataset.get_column("antibody_chain").to_list()
        chain_types = dataset.get_column("chain_type").to_list()
        labels = dataset.get_column("full_paratope_labels").to_list()

        if self.name == EndToEndPredictorName.paragraph_paired:
            heavys = dataset.filter(pl.col("chain_type").is_in(["heavy"]))
            lights = dataset.filter(pl.col("chain_type").is_in(["light"]))
            paired_inputs = heavys.join(lights, on="pdb").rename({"antibody_chain": "H", "antibody_chain_right": "L"})
            paired_inputs = paired_inputs.select(["pdb", "H", "L"])

            for pdb, ab_chain, chain_type in zip(pdbs, ab_chains, chain_types):
                inputs.append({
                    "pdb_code": [pdb],
                    "H_id": paired_inputs.filter(pl.col("pdb") == pdb).select("H").item(),
                    "L_id": paired_inputs.filter(pl.col("pdb") == pdb).select("L").item(),
                    "focus": ab_chain
                })
        else:
            for pdb, ab_chain, chain_type in zip(pdbs, ab_chains, chain_types):
                inputs.append({
                    "pdb_code": [pdb],
                    "H_id": ["" if chain_type == "light" else ab_chain],
                    "L_id": ["" if chain_type == "heavy" else ab_chain],
                    "focus": ab_chain
                })
        return inputs, labels


# noinspection PyPep8Naming
class PLMSKClassifier(Predictor):
    """
    A composition of a Protein Language Model in the format provided by embedders.Embedder and a sklearn classifier
    that outputs probabilities. In this class embeddings are provided "as is", i.e. no fine-tuning is done.
    """

    def __init__(
            self,
            embedder: embedders.Embedder | embedders.MultiChainAwareEmbedder,
            classifier: RandomForestClassifier | HistGradientBoostingClassifier,
            save_path: Path | str | None = None,
            de_bias: bool = False,
    ):
        self.name = embedder.name
        self.embedder = embedder
        self.classifier = classifier
        self.save_path = save_path
        self.de_bias = de_bias

    def train(self, X_train: list[str] | list[tuple[str, str]], y_train: list[list[bool]]):
        """
        Trains self.classifier on the embeddings computed with a given classifier.

        Parameters
        ----------
        X_train: list of str where each string is an antibody amino acidic sequence.
        y_train: list of boolean arrays that represent the labels corresponding to the paratope.

        """

        # First get the embedding from the embedder
        print(f"Computing embeddings for: {self.embedder.name}")
        X_train_embs = [self.embedder.embed(inputs)[0].to("cpu").numpy() for inputs in tqdm(X_train)]
        X_train_embs = np.vstack(X_train_embs).astype(float)
        y_train = np.concatenate(y_train)

        if self.de_bias:
            # One could theoretically de-bias by first clustering, in this case we know that the distribution of AAs
            # is tilted towards Y, i.e. Tyrosines. So we could hope to cluster by amino acid the embeddings and
            # equal out the classes.
            # Unfortunately, this does not work that well
            """
            print("Fitting clustering algorithm")
            kmeans = KMeans(n_clusters=20, random_state=42).fit(X_train_embs)
            _, counts_clust = np.unique(kmeans.labels_, return_counts=True)

            # median = np.median(counts_clust)
            # median_counts_clust_idx = np.argmin([median - counts for counts in counts_clust])

            X_train_embs_de_biased = []
            y_train_de_biased = []

            print("Calculating the new de-biased training set")
            for x, y in zip(X_train_embs, y_train):
                pred_clust = kmeans.predict([x])
                counts_clust[pred_clust] -= 1
                X_train_embs_de_biased.append(x)
                y_train_de_biased.append(y)
                # Break when the smallest cluster is consumed
                if counts_clust[pred_clust] == 0:
                    break
                # Break when the median cluster is consumed
                # if counts_clust[median_counts_clust_idx] == 0:
                #    break
            else:
                raise NotImplementedError("There was not a full digestion of at least one class")
            X_train_embs = X_train_embs_de_biased
            y_train = y_train_de_biased
            """

            # Let's use a hard de-biasing, handcrafted, without clustering.
            aas = np.array([aa for aa in "".join(X_train)])
            unique_aas, counts_aas = np.unique(aas, return_counts=True)
            aa_to_counts = {ua: ca for ua, ca in zip(unique_aas, counts_aas)}

            X_train_embs_de_biased = []
            y_train_de_biased = []

            print("Calculating the new de-biased training set")
            for x, aa, y in zip(X_train_embs, aas, y_train):
                aa_to_counts[aa] -= 1
                X_train_embs_de_biased.append(x)
                y_train_de_biased.append(y)
                # Break when the less frequent aa class is consumed.
                if aa_to_counts[aa] == 0:
                    break

            X_train_embs = X_train_embs_de_biased
            y_train = y_train_de_biased

        print(f"Fitting classification head: {self.classifier.__class__.__name__}")
        self.classifier.fit(X_train_embs, y_train)
        return self

    def predict(self, an_input: str | tuple[str, str]) -> np.ndarray:
        """
        Predicts the paratope by computing embeddings and running a trained classifier on top of them.

        Parameters
        ----------
        an_input: an amino acidic sequence

        Returns
        -------
        Predicted probabilities
        """
        emb = self.embedder.embed(an_input)[0].to("cpu").numpy()
        output = self.classifier.predict_proba(emb)[:, -1]
        return output.flatten()

    def predict_multiple(self, some_inputs: list[str] | list[tuple[str, str]]) -> list[np.ndarray]:
        return [self.predict(inputs) for inputs in some_inputs]

    def prepare_dataset(
            self,
            dataset: pl.DataFrame,
    ) -> tuple[list[str], list[list[bool]]]:
        """
        Prepares a given DataFrame to be consumed.

        Parameters
        ----------
        dataset: A DataFrame that has to contain the following columns: "antibody_sequence" and "full_paratope_labels"

        Returns
        -------
        inputs, labels: These are grouped by protein i.e. each element in the inputs is an antibody, and each element
                        in labels is list of labels for each amino acid of the antibody sequence in the inputs.
        """

        inputs = []
        pdbs = dataset.get_column("pdb").to_list()
        ab_seqs = dataset.get_column("antibody_sequence").to_list()
        ag_seqs = dataset.get_column("antigen_sequence").to_list()
        chain_types = dataset.get_column("chain_type").to_list()
        labels = dataset.get_column("full_paratope_labels").to_list()

        if "ag_aware" in self.name or "paired" in self.name:
            second_seqs = ab_seqs if "paired" in self.name else ag_seqs
            for pdb_i, main_seq, c_type_i in zip(pdbs, ab_seqs, chain_types):
                for pdb_j, context_seq, c_type_j in zip(pdbs, second_seqs, chain_types):
                    if pdb_i == pdb_j and c_type_i != c_type_j:
                        inputs.append(tuple([main_seq, context_seq]))
        else:
            inputs = ab_seqs
        return inputs, labels


class ContactsClassifier(Predictor):
    """
    Structure for protein complexes can be computed through the usage of AlphaFold Multimer or ESMFold.-
    These precomputed complexes can be labelled according to the SabdabHub.label_dimer, thus making them comparable to
    other predictors.
    """

    def __init__(self, name: EndToEndPredictorName.af2_multimer):
        self.name = name

    def predict(self, an_input: Path | str) -> np.ndarray:
        """
        Provides paratope labels to a predicted structure of a dimer.

        Parameters
        ----------
        an_input: a Path to a PDB file.

        Returns
        -------
        The array containing the computed residues that correspond to the paratope of the predicted structure.

        """
        df = SabdabHub.label_dimer(
            pdb_path=an_input,
            chain_1="A",
            chain_2="B",
            contact_threshold=4.5,
            invalid_chains={""},
        )
        paratope = df.select("paratope_labels").to_numpy().flatten()[0]
        return paratope.astype(int).astype(float)

    def predict_multiple(self, some_inputs: list[Path | str]) -> list[np.ndarray]:
        """

        Parameters
        ----------
        some_inputs: list of PDB file Paths.

        Returns
        -------
        The list of arrays containing the computed residues that correspond to the paratope of the predicted structure.
        """
        return [self.predict(an_input) for an_input in some_inputs]

    def prepare_dataset(
            self,
            dataset: pl.DataFrame,
    ) -> tuple[list[str], list[bool]]:
        """
        Prepares a given DataFrame to be consumed.
        This is a mock version of a true prepare_dataset, since it loads already computed
        pdb files from AF2 Multimer for example, for simplicity.
        Consider implementing a true forward pass into a multimer structure prediction model for future use cases.

        Parameters
        ----------
        dataset: A DataFrame that has to contain the following columns: "antibody_sequence" and "full_paratope_labels"

        Returns
        -------
        inputs, labels: These are grouped by protein i.e. each element in the inputs is an antibody, and each element
                        in labels is list of labels for each amino acid of the antibody sequence in the inputs.
        """

        inputs = []
        labels = dataset.get_column("full_paratope_labels").to_list()
        pdbs = dataset.get_column("pdb").to_list()
        chain_types = dataset.get_column("chain_type").to_list()

        # Overkill, but ensures the order is the same.
        # FIXME: find a proper way
        def get_pdb_idx(pdbs_paths, _pdb_name):
            for i, p_path in enumerate(pdbs_paths):
                if _pdb_name.upper() in p_path:
                    return i

        vhag_af = TOPEFIND_PATH / "resources/computed_structures_paragraph_test/vhag/AF_structures"
        vlag_af = TOPEFIND_PATH / "resources/computed_structures_paragraph_test/vlag/AF_structures"
        pdbs_paths_test_af_h = [str(p) for p in list(vhag_af.glob("*.pdb"))]
        pdbs_paths_test_af_l = [str(p) for p in list(vlag_af.glob("*.pdb"))]

        for pdb_name, chain_type in zip(pdbs, chain_types):
            try:
                if chain_type == "heavy":
                    inputs.append(pdbs_paths_test_af_h[get_pdb_idx(pdbs_paths_test_af_h, pdb_name)])
                else:
                    inputs.append(pdbs_paths_test_af_l[get_pdb_idx(pdbs_paths_test_af_l, pdb_name)])
            except TypeError as e:
                print(f"You might have not computed some files with AF2, missing: {pdb_name} {chain_type}")
                print(e)

        return inputs, labels


class Seq2ParatopeCDR(Predictor):
    def __init__(self, name: EndToEndPredictorName.seq_to_cdr):
        self.name = name

    def predict(self, an_input: str) -> np.ndarray:
        """
        Provides paratope labels by setting as paratope the CDR regions.

        Parameters
        ----------
        an_input: an amino acidic sequence of an antibody.

        Returns
        -------
        The array containing the computed residues that correspond to the paratope of the predicted structure.

        """
        cdrs, fmks = utils.get_antibody_regions(an_input, scheme="imgt")
        paratope = np.zeros(len(fmks[0]))
        for cdr, fmk in zip(cdrs, fmks[1:]):
            paratope = np.concatenate([paratope, np.ones(len(cdr)), np.zeros(len(fmk))])
        return paratope

    def predict_multiple(self, some_inputs: list[str]) -> list[np.ndarray]:
        """

        Parameters
        ----------
        some_inputs: list of PDB file Paths.

        Returns
        -------
        The list of arrays containing the computed residues that correspond to the paratope of the predicted structure.
        """
        return [self.predict(an_input) for an_input in some_inputs]

    def prepare_dataset(
            self,
            dataset: pl.DataFrame,
    ) -> tuple[list[str], list[bool]]:
        """
        Prepares a given DataFrame to be consumed.
        
        Parameters
        ----------
        dataset: A DataFrame that has to contain the following columns: "antibody_sequence" and "full_paratope_labels"

        Returns
        -------
        inputs, labels: These are grouped by protein i.e. each element in the inputs is an antibody, and each element
                        in labels is list of labels for each amino acid of the antibody sequence in the inputs.
        """

        inputs = dataset.get_column("antibody_sequence").to_list()
        labels = dataset.get_column("full_paratope_labels").to_list()
        return inputs, labels
