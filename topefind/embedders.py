"""
This implementation focuses on gathering known protein embedding methods.
Following the abstract base class, embedders should provide an embed method.
The responsibility of creating the subsystems falls under each concrete Embedder.
"""
import os
import re
import sys
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import pandas as pd
import polars as pl
import numpy as np

from transformers import AutoTokenizer

if sys.version_info >= (3, 11):
    from enum import StrEnum, auto
else:
    from backports.strenum import StrEnum
    from backports import auto

from topefind import utils
from topefind.utils import (
    VALID_IMGT_IDS,
    AAS_FEATURES_PATH,
    TOPEFIND_PATH,
    get_antibody_numbering,
    download_url,
)

FILE_PATH = Path(__file__)


class EmbedderName(StrEnum):
    rita_s = auto()
    rita_m = auto()
    rita_l = auto()
    rita_xl = auto()
    esm1b = auto()
    esm2_8m = auto()
    esm2_35m = auto()
    esm2_150m = auto()
    esm2_650m = auto()
    esm2_3b = auto()
    esm_fold = auto()
    prot_t5_xl = auto()
    prot_t5_xxl = auto()
    aa = auto()
    imgt = auto()
    imgt_aa = auto()
    imgt_aa_ctx_3 = auto()
    imgt_aa_ctx_5 = auto()
    imgt_aa_ctx_7 = auto()
    imgt_aa_ctx_11 = auto()
    imgt_aa_ctx_17 = auto()
    imgt_aa_ctx_23 = auto()


class RemoteEmbedderName(StrEnum):
    # Same names are maintained with the EmbedderName ones for ease of call by the ConfigurizedEmbedder.
    rita_s = "lightonai/RITA_s"
    rita_m = "lightonai/RITA_m"
    rita_l = "lightonai/RITA_l"
    rita_xl = "lightonai/RITA_xl"
    esm1b = "facebook/esm1b_t33_650M_UR50S"
    esm2_8m = "facebook/esm2_t6_8M_UR50D"
    esm2_35m = "facebook/esm2_t12_35M_UR50D"
    esm2_150m = "facebook/esm2_t30_150M_UR50D"
    esm2_650m = "facebook/esm2_t33_650M_UR50D"
    esm2_3b = "facebook/esm2_t36_3B_UR50D"
    esm_fold = "facebook/esmfold_v1"
    prot_t5_xl = "Rostlab/prot_t5_xl_uniref50"
    prot_t5_xxl = "Rostlab/prot_t5_xxl_uniref50"


class MultiChainType(StrEnum):
    ag_aware = auto()
    paired = auto()


class Embedder(ABC):
    """
    ABC for an embedder. Derived classes should "embed", and have a name.
    """
    name: EmbedderName

    @abstractmethod
    def embed(self, *inputs):
        ...


class ConfigurizedEmbedder:
    """
    In this implementation, a concrete embedder inherits from the abstraction of the embedder, but also from
    a ConfigurizedEmbedder. This allows for ease of usage and implementation of HuggingFace models in the same
    way (saves many lines of code).
    """

    def __init__(
            self,
            device_type: str,
            name: EmbedderName,
            tokenizer,
            model,
            tokenizer_kwargs,
            model_kwargs,
    ):
        self.name = name
        self.device = utils.get_device(device_type)
        model_url = f"https://huggingface.co/{RemoteEmbedderName[self.name]}"
        model_name_no_team = RemoteEmbedderName[self.name].split("/")[1]
        model_path = TOPEFIND_PATH.parent / "models" / model_name_no_team

        stdouts = []
        if not model_path.exists():
            print(f"Downloading model: {name}")
            cwd = os.getcwd()
            os.chdir(TOPEFIND_PATH.parent / "models")
            stdouts.append(subprocess.run(["git", "clone", f"{model_url}"], capture_output=True, text=True))
            weights_url = f"{model_url}/resolve/main/pytorch_model.bin"
            download_url(weights_url, model_path / "pytorch_model.bin", chunk_size=1024)
            os.chdir(cwd)
            [print(result.stdout) for result in stdouts]

        self.tokenizer = tokenizer.from_pretrained(
            model_path, **tokenizer_kwargs
        )
        self.model = model.from_pretrained(
            model_path, **model_kwargs
        )

        self.model.to(self.device)
        self.model.eval()


class MultiChainAwareEmbedder(Embedder):
    """
    To allow extra functionality, MultiChainAwareEmbedder allows the consideration of two chains.
    The first one is embedded with the functionality of the embedder, while the second one is concatenated by means of
    its average embedding.
    We favor composition, thus an Embedder must be provided.
    """

    def __init__(
            self,
            embedder: (Embedder, ConfigurizedEmbedder),
            multi_chain_type: MultiChainType,
    ):
        self.embedder = embedder
        self.name = embedder.name + f"_{multi_chain_type}"

    def embed(self, inputs: list[tuple[str, str]] | tuple[str, str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: inputs for the multi chains

        Returns
        -------
        A list of embeddings
        """
        outputs = []

        if isinstance(inputs, tuple):
            inputs = [inputs]

        for an_input in inputs:
            ch1_in = an_input[0]
            ch2_in = an_input[1]
            ch1_embs = self.embedder.embed(ch1_in)[0]
            ch2_embs = self.embedder.embed(ch2_in)[0]
            avg_ch2_emb = torch.mean(ch2_embs, dim=0).squeeze(0)
            broadcasted_avg_ag = torch.outer(torch.ones(len(ch1_embs)).to(self.embedder.device), avg_ch2_emb)
            ch1_ch2_emb = torch.cat((ch1_embs, broadcasted_avg_ag), dim=1)
            outputs.append(ch1_ch2_emb)

        return outputs


class ESMEmbedder(Embedder, ConfigurizedEmbedder):
    """
    ESMEmbedder provides an interface for the HuggingFace ESM models.
    Depending on the model's name a different model will be used.
    Follows the documentation from: https://huggingface.co/docs/transformers/model_doc/esm.
    These embedders are BERT-based.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.esm2_8m,
            device: str = "auto",
    ):
        from transformers import EsmForMaskedLM
        ConfigurizedEmbedder.__init__(self, device, name, AutoTokenizer, EsmForMaskedLM, {}, {})

        # Model specific extra configs
        self.model.config.output_hidden_states = True

    @torch.no_grad()
    def embed(self, inputs: list[str] | str) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """
        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            an_input = [an_input]
            an_input = self.tokenizer(an_input, return_tensors="pt", add_special_tokens=False)
            an_input = an_input.to(self.device)
            output = self.model(**an_input).hidden_states[-1].squeeze(0)
            outputs.append(output)

        return outputs


class ESMFoldEmbedder(Embedder, ConfigurizedEmbedder):
    """
    ESMFoldEmbedder provides an interface for the HuggingFace ESMFold model.
    This is necessary since the embeddings from ESMFold require a different handling for extraction.
    Follows the documentation from: https://huggingface.co/facebook/esmfold_v1.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.esm_fold,
            device: str = "auto",
    ):
        from transformers import EsmForProteinFolding
        ConfigurizedEmbedder.__init__(self, device, name, AutoTokenizer, EsmForProteinFolding, {}, {})

    @torch.no_grad()
    def embed(self, inputs: list[str] | str) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """
        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            an_input = [an_input]
            an_input = self.tokenizer(an_input, return_tensors="pt", add_special_tokens=False)
            an_input = an_input.to(self.device)
            output = self.model(**an_input).s_s[-1].squeeze(0)
            outputs.append(output)

        return outputs


class ProtT5Embedder(Embedder, ConfigurizedEmbedder):
    """
    ProtT5Embedder provides an interface to the HuggingFace ProtT5 models.
    Follows the documentation from https://huggingface.co/Rostlab/prot_t5_xl_uniref50.
    These embedders are T5 based.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.prot_t5_xl,
            device: str = "auto",
    ):
        from transformers import T5EncoderModel, T5Tokenizer
        ConfigurizedEmbedder.__init__(self, device, name, T5Tokenizer, T5EncoderModel,
                                      {"do_lower_case": True}, {})

    @torch.no_grad()
    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """
        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            seq_length = len(an_input)
            an_input = [an_input]
            # Replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            an_input = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in an_input]
            # Tokenize sequences and pad up to the longest sequence in the batch
            ids = self.tokenizer.batch_encode_plus(an_input, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids["input_ids"]).to(self.device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            output = output.last_hidden_state[0, :seq_length]
            outputs.append(output)

        return outputs


class RITAEmbedder(Embedder, ConfigurizedEmbedder):
    """
    RITAEmbedder provides an interface for the RITA models.
    Follows the documentation from: https://huggingface.co/lightonai/RITA_l.
    These embedders are GPT-based.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.rita_s,
            device: str = "auto",
    ):
        from transformers import AutoModelForCausalLM
        ConfigurizedEmbedder.__init__(self, device, name, AutoTokenizer, AutoModelForCausalLM, {},
                                      {"trust_remote_code": True, "revision": True})
        # {"trust_remote_code": True} to load it, but first you need to check the configuration file.
        # check https://huggingface.co/lightonai/RITA_l for more.

    @torch.no_grad()
    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """
        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            an_input = [an_input]
            inputs = self.tokenizer(an_input, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(self.device)
            output = self.model(**inputs).hidden_states.squeeze(0)
            outputs.append(output)

        return outputs


class PhysicalPropertiesNoPosEmbedder(Embedder):
    """
    PhysicalPropertiesNoPosEmbedder is an overly simplified embedder for amino acid properties.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.aa,
            device: str = "auto",
    ):
        mapping_arr = pl.read_csv(AAS_FEATURES_PATH).drop("aa_name").to_numpy()
        self.name = name
        self.device = device  # To match the rest of the models
        self.mapping_dict = {str(row[0]): np.array(row[1:], dtype=float) for row in mapping_arr}
        self.features_dim = len(self.mapping_dict["A"])

    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """

        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            an_input_with_pos = np.zeros((len(an_input), self.features_dim))
            for i, aa in enumerate(an_input):
                an_input_with_pos[i] = self.mapping_dict[aa]

            output = torch.FloatTensor(an_input_with_pos)  # To match the rest of the models
            outputs.append(output)

        return outputs


class PhysicalPropertiesPosEmbedder(Embedder):
    """
    PhysicalPropertiesPosEmbedder is an overly simplified embedder for amino acid properties with IMGT positioning.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.imgt_aa,
            device: str = "auto",
            imgts: list = VALID_IMGT_IDS,
            precomputed_imgts_df: pd.DataFrame = pd.DataFrame(),
    ):
        mapping_arr = pl.read_csv(AAS_FEATURES_PATH).drop("aa_name").to_numpy()
        self.imgts = imgts
        self.name = name
        self.device = device  # To match the rest of the models
        self.mapping_dict = {str(row[0]): np.array(row[1:], dtype=float) for row in mapping_arr}
        self.features_dim = len(self.mapping_dict["A"])
        self.precomputed_imgts_df = precomputed_imgts_df

    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """

        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            if len(self.precomputed_imgts_df) > 0:
                imgt_from_df = self.precomputed_imgts_df[self.precomputed_imgts_df["antibody_sequence"] == an_input]
                imgt_from_df = imgt_from_df["antibody_imgt"].values
                imgt_numbering = list(imgt_from_df[0])
            else:
                imgt_numbering = get_antibody_numbering(an_input, scheme="imgt")
            # Append as last feature the position in IMGT numbering.
            an_input_with_pos = np.zeros((len(an_input), self.features_dim + 1))
            for i, (aa, imgt_num) in enumerate(zip(an_input, imgt_numbering)):
                an_input_with_pos[i, :-1] = self.mapping_dict[aa]
                an_input_with_pos[i, -1] = self.imgts.index(imgt_num)

            output = torch.FloatTensor(an_input_with_pos)  # To match the rest of the models
            outputs.append(output)

        return outputs


class IMGTPosEmbedder(Embedder):
    """
    IMGTPosEmbedder is an overly simplified embedder for amino acid position only.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.imgt,
            device: str = "auto",  # To match the signature of the rest of the models.
            imgts: list = VALID_IMGT_IDS,
            precomputed_imgts_df: pd.DataFrame = pd.DataFrame(),
    ):
        self.imgts = imgts
        self.name = name
        self.precomputed_imgts_df = precomputed_imgts_df

    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """

        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            if len(self.precomputed_imgts_df) > 0:
                imgt_from_df = self.precomputed_imgts_df[self.precomputed_imgts_df["antibody_sequence"] == an_input]
                imgt_from_df = imgt_from_df["antibody_imgt"].values
                imgt_numbering = list(imgt_from_df[0])
            else:
                imgt_numbering = get_antibody_numbering(an_input, scheme="imgt")
            self.latest_numbering_ = imgt_numbering
            # Append as last feature the position in IMGT numbering.
            an_input_with_pos = np.zeros((len(an_input), 1))
            for i, imgt_num in enumerate(imgt_numbering):
                an_input_with_pos[i, -1] = self.imgts.index(imgt_num)

            output = torch.FloatTensor(an_input_with_pos)  # To match the rest of the models
            outputs.append(output)

        return outputs


class PhysicalPropertiesPosContextEmbedder(Embedder):
    """
    PhysicalPropertiesPosContextEmbedder is an overly simplified embedder for amino acid properties with
    IMGT positioning and concatenation of context embeddings for surrounding AAs.
    """

    def __init__(
            self,
            name: EmbedderName = EmbedderName.imgt_aa_ctx_3,
            device: str = "auto",
            padding: int = -1,
            imgts: list = VALID_IMGT_IDS,
            precomputed_imgts_df: pd.DataFrame = pd.DataFrame(),
    ):
        mapping_arr = pl.read_csv(AAS_FEATURES_PATH).drop("aa_name").to_numpy()
        self.imgts = imgts
        self.name = name
        self.device = device  # To match the rest of the models
        self.context = int(str(name).split("_")[-1])
        self.padding = padding
        self.mapping_dict = {str(row[0]): np.array(row[1:], dtype=float) for row in mapping_arr}
        self.features_dim = len(self.mapping_dict["A"])
        self.precomputed_imgts_df = precomputed_imgts_df
        self.latest_numbering_ = None

    def embed(self, inputs: str | list[str]) -> list[torch.tensor]:
        """

        Parameters
        ----------
        inputs: input sequence or input sequences to embed.

        Returns
        -------
        A list of embeddings
        """

        outputs = []

        if isinstance(inputs, str):
            inputs = [inputs]

        for an_input in inputs:
            if len(self.precomputed_imgts_df) > 0:
                imgt_from_df = self.precomputed_imgts_df[self.precomputed_imgts_df["antibody_sequence"] == an_input]
                imgt_from_df = imgt_from_df["antibody_imgt"].values
                imgt_numbering = list(imgt_from_df[0])
            else:
                imgt_numbering = get_antibody_numbering(an_input, scheme="imgt")
            self.latest_numbering_ = imgt_numbering

            # Append as last feature the position in IMGT numbering.
            an_input_with_pos = np.zeros((len(an_input), self.features_dim + 1))
            for i, (aa, imgt_num) in enumerate(zip(an_input, imgt_numbering)):
                an_input_with_pos[i, :-1] = self.mapping_dict[aa]
                an_input_with_pos[i, -1] = self.imgts.index(imgt_num)

            # Let's append the context
            # Each residue has its context features as well e.g.
            # in WYT, Y will have features(W), features(Y), features(T)
            # We pad with extra glycines and a pad token.
            an_input_with_context = np.zeros((an_input_with_pos.shape[0], an_input_with_pos.shape[1] * self.context))
            sliding_window = np.arange(self.context) - ((self.context - 1) // 2)

            for i, _ in enumerate(an_input_with_pos):
                new_feature = np.zeros(an_input_with_pos.shape[1] * self.context)
                for j, window_pos in enumerate(sliding_window):
                    curr_pos = i + window_pos
                    if curr_pos < 0 or curr_pos >= len(an_input):
                        new_feature[j * an_input_with_pos.shape[1]:(j + 1) * an_input_with_pos.shape[1]] = \
                            np.concatenate([self.mapping_dict["G"], np.array([self.padding])])
                    else:
                        new_feature[j * an_input_with_pos.shape[1]:(j + 1) * an_input_with_pos.shape[1]] = \
                            an_input_with_pos[curr_pos]
                an_input_with_context[i] = new_feature

            output = torch.FloatTensor(an_input_with_context)  # To match the rest of the models
            outputs.append(output)

        return outputs


def get_embedder_constructor(name: EmbedderName | str):
    if "_rf" in name:
        name = name.removesuffix("_rf")
    if "_ag_aware" in name or "_paired" in name:
        return MultiChainAwareEmbedder
    if "esm" in name:
        return ESMEmbedder
    if "rita" in name:
        return RITAEmbedder
    if "prot_t5" in name:
        return ProtT5Embedder
    if "aa" == name:
        return PhysicalPropertiesNoPosEmbedder
    if "imgt" == name:
        return IMGTPosEmbedder
    if "imgt_aa" == name:
        return PhysicalPropertiesPosEmbedder
    if "imgt_aa_ctx" in name:
        return PhysicalPropertiesPosContextEmbedder
    raise ValueError("Wrong name in input")
