import pandas as pd
import re


def format_pdb(pdb_file):
    '''
    Process pdb file into pandas df
    
    Original author: Alissa Hummer
    
    :param pdb_file: file path of .pdb file to convert
    :returns: df with atomic level info
    '''
    
    pd.options.mode.chained_assignment = None
    pdb_whole = pd.read_csv(pdb_file,header=None,delimiter='\t')
    pdb_whole.columns = ['pdb']
    pdb = pdb_whole[pdb_whole['pdb'].str.startswith('ATOM')]
    pdb['Atom_Num'] = pdb['pdb'].str[6:11].copy()
    pdb['Atom_Name'] = pdb['pdb'].str[11:16].copy()
    pdb['AA'] = pdb['pdb'].str[17:20].copy()
    pdb['Chain'] = pdb['pdb'].str[20:22].copy()
    pdb['Res_Num'] = pdb['pdb'].str[22:27].copy().str.strip()
    pdb['x'] = pdb['pdb'].str[27:38].copy()
    pdb['y'] = pdb['pdb'].str[38:46].copy()
    pdb['z'] = pdb['pdb'].str[46:54].copy()#
    pdb['Atom_type'] = pdb['pdb'].str[77].copy()
    pdb.drop('pdb',axis=1,inplace=True)
    pdb.replace({' ':''}, regex=True, inplace=True)
    pdb.reset_index(inplace=True)
    pdb.drop('index',axis=1,inplace=True)
    
    # remove H atoms from our data (interested in heavy atoms only)
    pdb = pdb[pdb['Atom_type']!='H']

    return pdb


def get_ordered_AA_3_letter_codes():
    '''
    '''
    AA_unique_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                       'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                       'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                       'SER', 'THR', 'TRP', 'TYR', 'VAL']
    return AA_unique_names


def get_normal_CDR_loop_start_and_end_vals(loop_num):
    '''
    These are the normal IMGT start and end res nums for each CDR loop
    
    :param loop_num: int 1, 2, or 3
    :return: two-element list containing str of start and end nums
    '''
    if loop_num == 1:
        start, end = "27", "38"
    elif loop_num == 2:
        start, end = "56", "65"
    elif loop_num == 3:
        start, end = "105", "117"
    else:
        raise ValueError("loop_num must be either integer 1, 2, or 3")
    return start, end


def get_normal_CDRplus2_loop_start_and_end_vals(loop_num):
    '''
    These are the normal IMGT start and end res nums for each CDR loop + 2 extra res
    
    :param loop_num: int 1, 2, or 3
    :return: two-element list containing str of start and end nums
    '''
    if loop_num == 1:
        start, end = "25", "40"
    elif loop_num == 2:
        start, end = "54", "67"
    elif loop_num == 3:
        start, end = "103", "119"
    else:
        raise ValueError("loop_num must be either integer 1, 2, or 3")
    return start, end


def get_normal_Fv_start_and_end_vals(heavy=False):
    '''
    These are the normal IMGT start and end res nums for the Fv region
    
    :return: two-element list containing str of start and end nums
    '''
    start = "1"
    end = "128" if heavy else "127"
    return start, end


def search_up_for_nearest(start, end, res_num_list):
    '''
    In case start num isn't found, search for nearest res that is still in loop
    
    :param start: str res num that normally indicates start of loop
    :param end: str res num that normally indicates end of loop
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: nearest res to original start of loop or None if not found
    '''
    start = int(start)
    end = int(end)
    new_start = None

    for res_num in range(start, end+1):
        if str(res_num) in res_num_list:
            new_start = str(res_num)
            break
        res_num_flexi_insertion = re.compile(str(res_num) + "[A-Z]")
        tmp_list = list(filter(res_num_flexi_insertion.match, res_num_list))
        try:
            new_start = tmp_list[0]
            break
        except IndexError:
            pass
        
    return new_start


def search_down_for_nearest(start, end, res_num_list):
    '''
    In case end num isn't found, search for nearest res that is still in loop
    
    :param start: str res num that normally indicates start of loop
    :param end: str res num that normally indicates end of loop
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: nearest res to original end of loop or None if not found
    '''
    rev_res_num_list = res_num_list.copy()
    rev_res_num_list.reverse()
    start = int(start)
    end = int(end)
    new_end = None
    
    for res_num in range(end, start-1, -1):
        if str(res_num) in rev_res_num_list:
            new_end = str(res_num)
            break
        res_num_flexi_insertion = re.compile(str(res_num)+"[A-Z]")
        tmp_list = list(filter(res_num_flexi_insertion.match, rev_res_num_list))
        try:
            new_end = tmp_list[0]
            break
        except IndexError:
            pass
        
    return new_end
