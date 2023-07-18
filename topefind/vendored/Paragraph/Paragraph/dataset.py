import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from scipy.spatial.distance import cdist

from pathlib import Path
from topefind.vendored.Paragraph.Paragraph.utils import *

# default values using in training
SEARCH_AREA = "IMGT_CDRplus2"
NEIGHBOUR_RADIUS = 10


class ParagraphDataset(Dataset):

    def __init__(self, pdb_H_L_csv: pd.DataFrame | str | Path, pdb_folder_path):
        self.pdb_H_L_csv = pdb_H_L_csv
        self.pdb_folder_path = pdb_folder_path

        if isinstance(pdb_H_L_csv, pd.DataFrame):
            self.df_key = pdb_H_L_csv
        else:
            self.df_key = pd.read_csv(self.pdb_H_L_csv, header=None, names=["pdb_code", "H_id", "L_id"])

    def __len__(self):
        return self.df_key.shape[0]

    def __getitem__(self, index):

        # read in data from csv
        pdb_code = self.df_key.iloc[index]["pdb_code"]
        H_id = self.df_key.iloc[index]["H_id"]
        L_id = self.df_key.iloc[index]["L_id"]

        # read in and process imgt numbered pdb file - keep all atoms
        pdb_path = os.path.join(self.pdb_folder_path, pdb_code + ".pdb")
        df = format_pdb(pdb_path)

        # set nan coors to be zero - we do this here and not in the original function call as
        # otherwise edges would be formed between missing residues
        coors = get_CDR_coors(df, H_id, L_id).float()
        coors[coors != coors] = 0
        feats = get_all_CDR_node_features(df, H_id, L_id).float()
        edges = get_CDR_edge_features(df, H_id, L_id).float()
        graph = (feats, coors, edges)

        # extras data that may be useful in further analysis
        df_CDR = get_Calpha_CDR_only_df(df, H_id, L_id)
        AAs = ['' if AA is np.nan else AA for AA in df_CDR["AA"].values.tolist()]
        AtomNum = ['' if num is np.nan else num for num in df_CDR["Atom_Num"].values.tolist()]
        chain = df_CDR["Chain"].values.tolist()
        chain_type = ["H" if ID == H_id else "L" for ID in chain]

        # catch 'None' values and convert to string - unable to have None values in batches
        # this happens when there is only one string present
        chain = [str(chain_id) for chain_id in chain]
        IMGT = df_CDR["Res_Num"].values.tolist()
        x = ['' if x is np.nan else x for x in df_CDR["x"].values.tolist()]
        y = ['' if y is np.nan else y for y in df_CDR["y"].values.tolist()]
        z = ['' if z is np.nan else z for z in df_CDR["z"].values.tolist()]
        extras = (pdb_code, AAs, AtomNum, chain, chain_type, IMGT, x, y, z)

        return graph, extras


# ---------------------------------
# Main feature extraction functions
# ---------------------------------


def get_CDR_coors(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Get CDR C-alpha atom coordinates
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 3) with x, y, z coors of each atom
    '''

    # get CDR C-alpha atoms only
    df_CDRs = get_Calpha_CDR_only_df(df, H_id, L_id, search_area=search_area)

    # ensure coors are numbers
    df_CDRs["x"] = df_CDRs["x"].astype(float)
    df_CDRs["y"] = df_CDRs["y"].astype(float)
    df_CDRs["z"] = df_CDRs["z"].astype(float)

    # get coors as tensor
    coors = torch.tensor(df_CDRs[["x", "y", "z"]].values)

    return coors


def get_CDR_edge_features(df, H_id, L_id, neighbour_radius=NEIGHBOUR_RADIUS, search_area=SEARCH_AREA):
    '''
    Get tensor form of adjacency matrix for all CDR C-alpha atoms
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :param neighbour_radius: max distance in Angstroms neighbours can be
    :returns: tensor (num_CDR_residues, num_CDR_residues, 1) adj matrix 
    '''

    xyz_arr = get_CDR_coors(df, H_id, L_id, search_area=search_area).numpy()

    # get distances
    dist_matrix = cdist(xyz_arr, xyz_arr, 'euclidean')
    dist_tensor = torch.tensor(dist_matrix)

    # create adjacency matrix from distance info
    adj_matrix = torch.where(dist_tensor <= neighbour_radius, 1, 0)

    # remove self loops - do I want to do this???  
    adj_matrix = adj_matrix.fill_diagonal_(0, wrap=False)

    # adjust dimensions for model input
    adj_matrix.unsqueeze_(-1)

    return adj_matrix


def get_all_CDR_node_features(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Get tensor features embedding Amino Acid type and corresponding chain
    for each C-alpha atom in the CDR
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 76||26||22) with multi-hot encoding of selection from
              AA type (20), chain H/L (2), loop L1/.../H3 (6), and imgt num (54)
    '''

    return torch.cat((get_CDR_AA_onehot_features(df, H_id, L_id, search_area=search_area),
                      get_CDR_chain_onehot_features(df, H_id, L_id, search_area=search_area)), 1)


# ----------------
# Helper functions
# ----------------


def get_all_atom_CDR_df(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Create df containing data for all CDR heavy atoms
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: df with same cols as input but only rows for CDR atoms
    '''

    if search_area == "IMGT_CDR":
        H_res = get_all_present_CDR_resnum(df, H_id)
        L_res = get_all_present_CDR_resnum(df, L_id)
    elif search_area == "IMGT_CDRplus2":
        H_res = get_all_present_CDRplus2_resnum(df, H_id)
        L_res = get_all_present_CDRplus2_resnum(df, L_id)
    else:
        raise ValueError("Unexpected search_area string value")

    # define df with all possible chain type + number combos
    H_CDRs = [["H", H_id, Res_Num] for Res_Num in H_res]
    L_CDRs = [["L", L_id, Res_Num] for Res_Num in L_res]
    df_all_combos = pd.DataFrame(H_CDRs + L_CDRs, columns=["Chain_type", "Chain", "Res_Num"])

    # trim df so it contains only CDR residues that exist
    df_all_atom_CDRs = df[(df["Res_Num"].isin(list(set(H_res + L_res)))) &
                          (df["Chain"].isin([H_id, L_id]))]

    # left join on all combos so that we have rows for everything and NaNs where residue not present
    df_all_atom_CDRs = pd.merge(df_all_combos, df_all_atom_CDRs, how='left',
                                left_on=["Chain", "Res_Num"], right_on=["Chain", "Res_Num"])

    # drop duplicates e.g. where multiple NMR models exist
    df_all_atom_CDRs = df_all_atom_CDRs.drop_duplicates(subset=["Chain", "Res_Num", "Atom_Name"],
                                                        keep="first").reset_index(drop=True)

    return df_all_atom_CDRs


def get_Calpha_CDR_only_df(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Create smaller df containing only data for CDR C-alpha atoms
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: df with same cols as input but only rows for CDR C-alphas
    '''
    df_all_atom = get_all_atom_CDR_df(df, H_id, L_id, search_area)
    return df_all_atom[(df_all_atom["Atom_Name"].str.strip() == "CA")].reset_index(drop=True)


def get_CDR_AA_onehot_features(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Encodes CDR residues types as one-hot vectors for model input
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 20) one-hot encoding for each 20 AA types
    '''

    # get CDR C-alpha atoms only
    df_CDRs = get_Calpha_CDR_only_df(df, H_id, L_id, search_area=search_area)

    AA_unique_names = get_ordered_AA_3_letter_codes()
    AA_name_dict = {name: idx for idx, name in enumerate(AA_unique_names)}

    # nice names to make rest of code more understandable
    num_rows = df_CDRs.shape[0]
    num_AA = len(AA_unique_names)

    # convert AA name to one-hot encoding
    AA_onehot_matrix = np.zeros((num_rows, num_AA))

    # we will only non-zero elements where residues actually exist
    df_CDRs_not_null = df_CDRs[~df_CDRs["AA"].isna()]
    df_CDRs_not_null_indices = df_CDRs_not_null.index.values

    AA_onehot_matrix[df_CDRs_not_null_indices,
    [AA_name_dict[residue] for residue in df_CDRs_not_null["AA"]]] = 1

    # convert from numpy to tensor
    AA_onehot_tensor = torch.tensor(AA_onehot_matrix)

    return AA_onehot_tensor


def get_CDR_chain_onehot_features(df, H_id, L_id, search_area=SEARCH_AREA):
    '''
    Encodes chain corresponding to each CDR residue as one-hot vectors for model input
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 2) one-hot encoding for H vs L chain
    '''

    # get CDR C-alpha atoms only
    df_CDRs = get_Calpha_CDR_only_df(df, H_id, L_id, search_area=search_area)

    chain_name_dict = {"H": 0, "L": 1}

    # nice names to make rest of code more understandable
    num_rows = df_CDRs.shape[0]
    num_chain_types = 2

    # convert chain type to one-hot encoding    
    chain_onehot_matrix = np.zeros((num_rows, num_chain_types))

    # we will only non-zero elements where residues actually exist
    df_CDRs_not_null = df_CDRs[~df_CDRs["AA"].isna()]
    df_CDRs_not_null_indices = df_CDRs_not_null.index.values

    chain_onehot_matrix[df_CDRs_not_null_indices,
    [chain_name_dict[residue] for residue in df_CDRs_not_null["Chain_type"]]] = 1

    # convert from numpy to tensor
    chain_onehot_tensor = torch.tensor(chain_onehot_matrix)

    return chain_onehot_tensor


def get_ordered_seq_from_df(df, chainID):
    '''
    Get the full ordered amino acid seq for a protein chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: ordered list of str of all res nums in certain chain
    '''
    df_Calpha_chain_of_interest = df[(df["Chain"] == chainID) & (df["Atom_Name"] == "CA")]
    return df_Calpha_chain_of_interest["Res_Num"].values.tolist()


def get_present_CDR_loop_start_and_end_vals(loop_num, res_num_list):
    '''
    These are the adapted IMGT start and end res nums for each CDR loop
    
    :param loop_num: int 1, 2, or 3
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: two-element list containing str of start and end nums
    '''
    og_start, og_end = get_normal_CDR_loop_start_and_end_vals(loop_num)

    start = og_start if og_start in res_num_list else search_up_for_nearest(og_start, og_end, res_num_list)
    end = og_end if og_end in res_num_list else search_down_for_nearest(og_start, og_end, res_num_list)

    return start, end


def get_all_present_CDR_resnum(df, chainID):
    '''
    Get ordered res nums present in CDR loops only for single chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: list of str of res nums in single chain CDR loops
    '''
    all_present_resnums = get_ordered_seq_from_df(df, chainID)
    CDR_loops = []

    for loop_num in range(1, 4):

        start, end = get_present_CDR_loop_start_and_end_vals(loop_num, all_present_resnums)
        try:
            loop_residues = all_present_resnums[all_present_resnums.index(start):all_present_resnums.index(end) + 1]
            CDR_loops = loop_residues if CDR_loops == [] else CDR_loops + loop_residues
        except ValueError:
            pass  # catches when start / end is None

    return CDR_loops


def get_present_CDRplus2_loop_start_and_end_vals(loop_num, res_num_list):
    '''
    These are the adapted IMGT start and end res nums for each CDR loop + 2 extra res
    Can't guarantee there aren't weird insertions e.g. 26A, 26B, so we step carefully
    We assume that any missing residues are simply not imaged/numbered correctly, and so
    avoid taking steps beyond the normal target e.g. will allow 119, but not 120
    
    :param loop_num: int 1, 2, or 3
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: two-element list containing str of start and end nums
    '''
    # normal CDR + 2 loop start / end
    og_start, og_end = get_normal_CDR_loop_start_and_end_vals(loop_num)

    # normal CDR + 2 loop start / end
    og_start_plus2, og_end_plus2 = get_normal_CDRplus2_loop_start_and_end_vals(loop_num)

    # initiate start and end as present CDR start and end
    start, end = get_present_CDR_loop_start_and_end_vals(loop_num, res_num_list)

    try:
        # find CDR loop start and end
        idx_CDR_loop_start = res_num_list.index(start)
        idx_CDR_loop_end = res_num_list.index(end)

        # step twice away from the loop ends
        for count in range(1, 3):

            # left
            proposed_CDRplus2_start = res_num_list[max(0, idx_CDR_loop_start - count)]
            proposed_CDRplus2_start_num = int(re.findall("[0-9]+", proposed_CDRplus2_start)[0])
            if proposed_CDRplus2_start_num >= int(og_start_plus2):
                start = proposed_CDRplus2_start

            # right
            proposed_CDRplus2_end = res_num_list[min(len(res_num_list) - 1, idx_CDR_loop_end + count)]
            proposed_CDRplus2_end_num = int(re.findall("[0-9]+", proposed_CDRplus2_end)[0])
            if proposed_CDRplus2_end_num <= int(og_end_plus2):
                end = proposed_CDRplus2_end

    except ValueError:
        pass  # when CDR loop start / end is not found

    return start, end


def get_all_present_CDRplus2_resnum(df, chainID):
    '''
    Get ordered res nums present in CDR loops + 2 only for single chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: list of str of res nums in single chain CDR loops + 2
    '''
    all_present_resnums = get_ordered_seq_from_df(df, chainID)
    CDRplus2_loops = []

    for loop_num in range(1, 4):

        start, end = get_present_CDRplus2_loop_start_and_end_vals(loop_num, all_present_resnums)
        try:
            loop_residues = all_present_resnums[all_present_resnums.index(start):all_present_resnums.index(end) + 1]
            CDRplus2_loops = loop_residues if CDRplus2_loops == [] else CDRplus2_loops + loop_residues
        except ValueError:
            pass  # catches when start / end is None

    return CDRplus2_loops


def get_present_Fv_start_and_end_vals(res_num_list, heavy=False):
    '''
    These are the adapted IMGT start and end res nums for the Fv region
    
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: two-element list containing str of start and end nums
    '''
    og_start, og_end = get_normal_Fv_start_and_end_vals(heavy=heavy)

    start = og_start if og_start in res_num_list else search_up_for_nearest(og_start, og_end, res_num_list)
    end = og_end if og_end in res_num_list else search_down_for_nearest(og_start, og_end, res_num_list)

    return start, end


def get_all_present_Fv_resnum(df, chainID, heavy=False):
    '''
    Get ordered res nums present in Fv region only for single chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: list of str of res nums in single chain Fv
    '''
    all_present_resnums = get_ordered_seq_from_df(df, chainID)
    Fv = []

    start, end = get_present_Fv_start_and_end_vals(all_present_resnums, heavy=heavy)
    try:
        Fv = all_present_resnums[all_present_resnums.index(start):all_present_resnums.index(end) + 1]
    except ValueError:
        pass  # catches when start / end is None

    return Fv
