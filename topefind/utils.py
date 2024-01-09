import os
import re
from pathlib import Path
from typing import Callable

import requests
import torch
import anarci
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
)
from tqdm import tqdm

# GLOBALS
SPECIAL_CHAR_ANARCI = "-"

TOPEFIND_PATH = Path(__file__).parent
SABDAB_PATH = TOPEFIND_PATH.parent / "datasets" / "sabdab"
MODELS_PATH = TOPEFIND_PATH.parent / "models"
AAS_FEATURES_PATH = TOPEFIND_PATH.parent / "resources/aas_features.csv"
AAs = np.array([aa for aa in "ACDEFGHIKLMNPQRSTVWY"])

# Easily expandable to bigger CDR3s:
# Check: https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html for more.
# The following follows ANARCI output convention with e.g. 111A for 111.1 and so on.
VALID_IMGT_CDR3_IDS = [
    "111A", "111B", "111C", "111D", "111E", "111F", "111G", "111H", "111I",
    "112I", "112H", "112G", "112F", "112E", "112D", "112C", "112B", "112A",
]
VALID_IMGT_IDS = [str(i) for i in range(1, 112)] + VALID_IMGT_CDR3_IDS + [str(i) for i in range(112, 129)]
VALID_SET_IMGT_IDS = set(VALID_IMGT_IDS)

# Regions defined by parapred paper: -2 start, +2 end
PARAPRED_REGIONS_CHOTHIA = {
    "H": {
        "CDR1": np.arange(24, 37),
        "CDR2": np.arange(50, 59),
        "CDR3": np.arange(93, 105),
        "FMK1": np.arange(0, 24),
        "FMK2": np.arange(37, 50),
        "FMK3": np.arange(59, 93),
    },
    "L": {
        "CDR1": np.arange(22, 37),
        "CDR2": np.arange(48, 59),
        "CDR3": np.arange(87, 100),
        "FMK1": np.arange(0, 22),
        "FMK2": np.arange(37, 48),
        "FMK3": np.arange(59, 87),
    },
}

PARAPRED_REGIONS_IMGT = {
    "H": {
        "CDR1": np.arange(25, 40),
        "CDR2": np.arange(55, 67),
        "CDR3": np.arange(105, 120),
        "FMK1": np.arange(0, 25),
        "FMK2": np.arange(40, 55),
        "FMK3": np.arange(67, 100),
    },
    "L": {
        "CDR1": np.arange(22, 43),
        "CDR2": np.arange(54, 72),
        "CDR3": np.arange(103, 119),
        "FMK1": np.arange(0, 22),
        "FMK2": np.arange(43, 54),
        "FMK3": np.arange(72, 103),
    },
}

REGIONS_IMGT = {
    "H": {
        "CDR1": np.arange(27, 39),
        "CDR2": np.arange(56, 66),
        "CDR3": np.arange(105, 118),
        "FMK1": np.arange(0, 27),
        "FMK2": np.arange(39, 56),
        "FMK3": np.arange(66, 105),
    },
    "L": {
        "CDR1": np.arange(27, 39),
        "CDR2": np.arange(56, 66),
        "CDR3": np.arange(105, 118),
        "FMK1": np.arange(0, 27),
        "FMK2": np.arange(39, 56),
        "FMK3": np.arange(66, 105),
    },
}

TOP_KS = (3, 5, 10, 15)

METRICS_NAMES = [
    "ap",
    "roc_auc",
    "mcc",
    "iou",
    "precision",
    "recall",
    "f1",
    "bal_acc",
    "aiou",
    "prec@3",
    "prec@5",
    "prec@10",
    "prec@15",
]

AB_REGIONS = [
    "all",
    "CDR1",
    "CDR2",
    "CDR3",
    "FMK1",
    "FMK2",
    "FMK3",
    "FMK4",
]

NAME_BEAUTIFIER = {
    "parapred": "Parapred",
    "paragraph_untrained": "Paragraph Untrained",
    "paragraph_unpaired": "Paragraph Unpaired",
    "paragraph_paired": "Paragraph Paired",
    "af2_multimer": "AF2M",
    "seq_to_cdr": "SEQ2CDR",
    "aa_freq": "AA Freq",
    "pos_freq": "Pos Freq",
    "aa_pos_freq": "AA + Pos Freq",
    "esm1b_rf": "ESM1b + RF",
    "esm2_8m_rf": "ESM2 8M + RF",
    "esm2_35m_rf": "ESM2 35M + RF",
    "esm2_150m_rf": "ESM2 150M + RF",
    "esm2_650m_rf": "ESM2 650M + RF",
    "esm2_3b_rf": "ESM2 3B + RF",
    "rita_s_rf": "RITA S + RF",
    "rita_m_rf": "RITA M + RF",
    "rita_l_rf": "RITA L + RF",
    "rita_xl_rf": "RITA XL + RF",
    "prot_t5_xl_rf": "ProtT5 XL + RF",
    "prot_t5_xxl_rf": "ProtT5 XXL + RF",
    "aa_rf": "AA + RF",
    "imgt_rf": "Pos + RF",
    "imgt_aa_rf": "AA + Pos + RF",
    "imgt_aa_ctx_3_rf": "AA + Pos + 3CTX + RF",
    "imgt_aa_ctx_5_rf": "AA + Pos + 5CTX + RF",
    "imgt_aa_ctx_7_rf": "AA + Pos + 7CTX + RF",
    "imgt_aa_ctx_11_rf": "AA + Pos + 11CTX + RF",
    "imgt_aa_ctx_17_rf": "AA + Pos + 17CTX + RF",
    "imgt_aa_ctx_23_rf": "AA + Pos + 23CTX + RF",
    "esm1b_paired_rf": "ESM1b + Paired + RF",
    "esm2_8m_paired_rf": "ESM2 8M + Paired + RF",
    "esm2_35m_paired_rf": "ESM2 35M + Paired + RF",
    "esm2_150m_paired_rf": "ESM2 150M + Paired + RF",
    "esm2_650m_paired_rf": "ESM2 650M + Paired + RF",
    "esm2_3b_paired_rf": "ESM2 3B + Paired + RF",
    "rita_s_paired_rf": "RITA S + Paired + RF",
    "rita_m_paired_rf": "RITA M + Paired + RF",
    "rita_l_paired_rf": "RITA L + Paired + RF",
    "rita_xl_paired_rf": "RITA XL + Paired + RF",
    "prot_t5_xl_paired_rf": "ProtT5 XL + Paired + RF",
    "prot_t5_xxl_paired_rf": "ProtT5 XXL + Paired + RF",
    "esm1b_ag_aware_rf": "ESM1b + AgAware + RF",
    "esm2_8m_ag_aware_rf": "ESM2 8M + AgAware + RF",
    "esm2_35m_ag_aware_rf": "ESM2 35M + AgAware + RF",
    "esm2_150m_ag_aware_rf": "ESM2 150M + AgAware + RF",
    "esm2_650m_ag_aware_rf": "ESM2 650M + AgAware + RF",
    "esm2_3b_ag_aware_rf": "ESM2M 3B + AgAware + RF",
    "rita_s_ag_aware_rf": "RITA S + AgAware + RF",
    "rita_m_ag_aware_rf": "RITA M + AgAware + RF",
    "rita_l_ag_aware_rf": "RITA L + AgAware + RF",
    "rita_xl_ag_aware_rf": "RITA XL + AgAware + RF",
    "prot_t5_xl_ag_aware_rf": "ProtT5 XL + AgAware + RF",
    "prot_t5_xxl_ag_aware_rf": "ProtT5 XXL + AgAware + RF",
    "esm2_8m_de_biased_rf": "ESM2 8M + DeBiased + RF",
    "esm2_35m_de_biased_rf": "ESM2 35M + DeBiased + RF",
    "esm2_150m_de_biased_rf": "ESM2 150M + DeBiased + RF",
    "esm2_650m_de_biased_rf": "ESM2 650M + DeBiased + RF",
    "esm2_3b_de_biased_rf": "ESM2 3B + DeBiased + RF",
    "esm2_8m_untrained_rf": "ESM2 8M w Random Weights + RF",
    "esm2_650m_untrained_rf": "ESM2 650M w Random Weights + RF",
}


# UTILITIES
def natsort(a_list):
    return sorted(a_list, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])


def rescale(x: np.ndarray):
    _min = np.min(x)
    _max = np.max(x)
    return (x - _min) / (_max - _min)


def metric_at_top_k(y_true, y_pred, k, metric=average_precision_score, thr=0.5):
    if k > len(y_true):
        k = len(y_true)
    top_k_ids = np.argpartition(y_pred, -k)[-k:]
    top_k_trues = y_true[top_k_ids].astype(int)
    top_k_preds = y_pred[top_k_ids]
    if metric != average_precision_score:
        top_k_preds = np.where(top_k_preds >= thr, 1, 0)
        return metric(top_k_trues, top_k_preds, zero_division=0)
    else:
        return metric(top_k_trues, top_k_preds)


def iou(y_true: np.ndarray, y_pred: np.ndarray):
    # If the union is 0 it means that yt = 0,0,...,0 and yp = 0,0,...,0, thus we set it to 1.
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union if union != 0 else 1


# Two ideas if you do not want to calibrate the models and have a sense of a metric
# that takes un-calibration into account.
def aiou(y_true: np.ndarray, y_pred: np.ndarray, n_thr=50):
    thrs = np.linspace(0.01, 0.99, n_thr)
    return np.mean([iou(y_true, np.where(y_pred < thr, 0, 1)) for thr in thrs])


def amcc(y_true: np.ndarray, y_pred: np.ndarray, n_thr=50):
    thrs = np.linspace(0.01, 0.99, n_thr)
    return np.mean([matthews_corrcoef(y_true, np.where(y_pred < thr, 0, 1)) for thr in thrs])


def find_free_device() -> str:
    if torch.cuda.is_available():
        os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > devices")
        with open("devices", "r") as file:
            memory = [int(x.split()[2]) for x in file.readlines()]
        device = f"cuda:{int(np.argmax(memory))}"
        os.remove("devices")
    else:
        device = "cpu"
    print(f"Using {device} device")
    return device


def get_device(mode: str = "auto") -> str:
    return find_free_device() if mode == "auto" else mode


def pad_to_imgt(
        arr_to_pad: np.ndarray[str | int],
        antibody_imgt: np.ndarray[str],
        available_imgt_ids: np.ndarray[str] = np.array(VALID_IMGT_IDS),
        max_imgt_pos: str | None = "128",
) -> np.ndarray:
    """
    This function has to pad a given input array to match the available_imgt_ids.

    Parameters
    ----------
    arr_to_pad: an array to pad, this can be for example an array of predictions, an array of labels or an array of
                amino acids that corresponds to the sequence.
    antibody_imgt: an array that corresponds to the available IMGT of the current arr_to_pad antibody.
    available_imgt_ids: this is a stored ordered set of IMGT ids which has to match to all available imgt in the dataset
                        at hand. Please recompute it if your dataset contains IMGT ids not present here.
    max_imgt_pos: the maximum wanted IMGT position. Sometimes, after CDR3 one might have a very long framework region
                  in IMGT pdbs from SAbDab for example. There might be the wish to clip it to a certain maximum.
    Returns
    -------
    A padded array.
    """

    pad_id = "-" if isinstance(arr_to_pad[0], str) else -100
    padded = np.full_like(arr_to_pad, shape=len(available_imgt_ids), fill_value=pad_id)

    un_mask = np.isin(available_imgt_ids, antibody_imgt)
    padded[un_mask] = arr_to_pad

    if max_imgt_pos:
        padded = padded[:np.argwhere(available_imgt_ids == max_imgt_pos)[0][0]]
    return padded


def get_imgt_region_indexes(
        ab_imgt: list[str],
        region: str,
        scheme: str = "imgt"
) -> np.ndarray:
    if scheme == "imgt":
        region_scheme_dict = REGIONS_IMGT
    else:
        raise NotImplementedError(f"Currently, {scheme} scheme is not implemented")

    indexes = []
    if region == "all":
        indexes = np.arange(len(ab_imgt))
    elif region == "FMK4":
        for idx, imgt_id in enumerate(ab_imgt):
            if int(re.sub(r"\D", "", imgt_id)) in region_scheme_dict["H"]["CDR3"]:
                indexes.append(idx)
        indexes = np.arange(len(ab_imgt))[indexes[-1] + 1:]
    elif region in ["CDR1", "CDR2", "CDR3", "FMK1", "FMK2", "FMK3"]:
        for idx, imgt_id in enumerate(ab_imgt):
            if int(re.sub(r"\D", "", imgt_id)) in region_scheme_dict["H"][region]:
                indexes.append(idx)
    else:
        raise ValueError(f"Wrong region, provide region in: {AB_REGIONS}")

    return np.array(indexes, dtype=int)


def get_antibody_numbering(
        sequence: str,
        scheme: str = "imgt",
        valid_numbering: set | None = None
) -> list[str] | None:
    """
   Wrapper function around ANARCI to provide antibody numbering.

   Parameters
   ----------
   sequence: Amino acid sequence of an antibody in string form.
   scheme: Antibody numbering scheme, tested ones are `imgt` or `chothia`.
           Valid schemes extend to all the ones allowed by ANARCI (not tested).
   valid_numbering: Set of valid positions/ids to include.

   Returns
   -------
   res_ins: residues accompanied by insertion codes in the style of ANARCI.

   """
    if valid_numbering is None:
        valid_numbering = VALID_SET_IMGT_IDS

    anarci_out, chain_type = anarci.number(sequence, scheme=scheme)
    if anarci_out is None or not anarci_out:
        return None

    anarci_out = np.array([[el[0][0], el[0][1], el[1]] for el in anarci_out])
    res_ids = anarci_out[:, 0].astype(int)
    res_ins = anarci_out[:, 1].astype(str)
    res_lets = anarci_out[:, 2].astype(str)
    res_ins = [i if i != " " else "" for i in res_ins]

    res_ins = [
        f"{r_id}{r_ins}"
        for r_id, r_ins, r_let in zip(res_ids, res_ins, res_lets)
        if r_let != SPECIAL_CHAR_ANARCI
    ]
    # Accounting for longer sequences to maintain the extra framework.
    if len(sequence) != len(res_ins):
        last_imgt = int(res_ins[-1])
        for i in range(len(res_ins) - len(res_ins)):
            res_ins.append(str(last_imgt + i + 1))

    for rid in res_ins:
        if rid not in valid_numbering:
            raise ValueError(f"ANARCI found a strange residue {rid}")
    return res_ins


def get_antibody_regions(
        sequence: str,
        scheme: str = "chothia"
) -> tuple[list, list]:
    """
    Wrapper function around ANARCI to provide antibody regions.

    Parameters
    ----------
    sequence: Amino acid sequence of an antibody in string form.
    scheme: Antibody numbering scheme, tested ones are `imgt` or `chothia`.
            Valid schemes extend to all the ones allowed by ANARCI (not tested).

    Returns
    -------
    regions: a tuple of two list `cdrs`, and `fmks`.
             `cdrs` is a list containing strings of found CDR regions.
             `fmks` is a list containing strings of found framework regions.
    """
    anarci_out, chain_type = anarci.number(sequence, scheme=scheme)

    # De-spaghetting anarci output into [res_id, ins_code, res_let]
    anarci_out = np.array([[el[0][0], el[0][1], el[1]] for el in anarci_out])
    res_ids = anarci_out[:, 0].astype(int)
    res_lets = anarci_out[:, 2].astype(str)

    # Getting the regions of interest
    regions = PARAPRED_REGIONS_CHOTHIA[chain_type] if scheme == "chothia" else REGIONS_IMGT[chain_type]
    indexes = {region: np.isin(res_ids, regions[region]) for region in regions}

    cdrs = [res_lets[indexes[region]] for region in regions if "CDR" in region]
    fmks = [res_lets[indexes[region]] for region in regions if "FMK" in region]

    cdrs = [list(filter(lambda aa: aa != SPECIAL_CHAR_ANARCI, cdr)) for cdr in cdrs]
    fmks = [list(filter(lambda aa: aa != SPECIAL_CHAR_ANARCI, fmk)) for fmk in fmks]

    length_cdrs = sum([len(r) for r in cdrs])
    length_fmks = sum([len(r) for r in fmks])

    length_regs = length_cdrs + length_fmks

    # Matching initial sequence length
    fmks.append([aa for aa in sequence[length_regs:]])
    return cdrs, fmks


def handle_naninf(arr: np.ndarray) -> np.ndarray:
    arr = np.where(np.isnan(arr), 0, arr)
    arr = np.where(np.isinf(arr), np.finfo(np.float64).max, arr)
    return arr


def kl_divergence(
        p: np.ndarray,
        q: np.ndarray,
        log: Callable = np.log2
) -> float:
    """
    Kullback-Leibler Divergence between two probability distributions.

    Parameters:
    p, q: Two probability distributions (must sum to 1 and be the same length).

    Returns: Kullback-Leibler Divergence.
    """

    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    return float(np.sum(p * log(p / q)))


def jensen_shannon_divergence(
        p: np.ndarray,
        q: np.ndarray
) -> float:
    """
    Jensen-Shannon Divergence between two probability distributions.

    Parameters:
    p, q: Two probability distributions (must sum to 1 and be the same length).

    Returns: Jensen-Shannon Divergence
    """

    p = handle_naninf(p)
    q = handle_naninf(q)

    # Get midpoint between the distributions
    m = (p + q) / 2

    # Compute Jensen-Shannon divergence
    divergence = (kl_divergence(p, m) + kl_divergence(q, m)) / 2

    return divergence


def download_url(
        url: str,
        save_path: Path | str,
        chunk_size: int = 128
):
    """
    Download a file from a given URL and show a progress bar.

    Parameters
    ----------
    url: The URL of the file to be downloaded.
    save_path: The local path where the file will be saved.
    chunk_size: The amount of data that should be read into memory at once, by default 128.
    """

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(chunk_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
