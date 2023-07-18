"""
Ground truth for 6qig.pdb
Computed contacts at 4.5Å between all heavy atoms.
Residues under 4.5Å are thus considered contacts.

H 28   PHE F C
H 29   THR T C
H 35   SER S C
H 37   TYR Y C
H 57   SER S C
H 58   SER S C
H 59   GLY G C
H 62   GLY G C
H 63   THR T C
H 64   TYR Y C
H 66   TYR Y C
H 82   ASN N C
H 107  ARG R C
H 110  TRP W C
H 111  ASP D C
H 111A 111A 111A C
H 111B 111B 111B C
H 112C 112C 112C C
H 112A 112A 112A C
H 113  TYR Y C
"""

from pathlib import Path

import numpy as np
import pandas as pd

from topefind.data_hub import SabdabHub

FILE_PATH = Path(__file__)
MOCK_SABDAB_PDBS_PATH = FILE_PATH.parent / "mock_data"


def test_label_dimer():
    pdb_path = MOCK_SABDAB_PDBS_PATH / "imgt" / "6qig.pdb"
    df = SabdabHub.label_dimer(pdb_path, "H", "A", 4.5, {"", "NA"}).to_pandas()

    paratope_mask = df["paratope_labels"].values[0]
    paratope_imgts = df["antibody_imgt"].values[0][paratope_mask]

    expected_paratope_imgts = np.array([
        "28", "29", "35", "37", "57", "58", "59", "62", "63", "64", "66",
        "82", "107", "110", "111", "111A", "111B", "112C", "112A", "113",
    ])

    assert np.all(paratope_imgts == expected_paratope_imgts)

    assert df["antibody_sequence"].values[0] == "EVQLVESGGDLVKPGGSLKLSCAASGFTFSSYGMSWVRQTPDKRLEWVATISSGGTYTYYADT" \
                                                "VKGRFTISRDNAKNNLYLQMSSLTSEDSAMFYCARRVAWDHGSTYDYAMDYWGQGTTVTVSSA" \
                                                "KTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTL" \
                                                "SSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDC"

    # The real antigen of this PDB has missing residues.
    # The missing region is not considered.
    # There should be a better way to do this (recomputing, filtering, etc...)!
    assert df["antigen_sequence"].values[0] == "ILHLELLVAVGPDVFQAHQEDTERYVLTNLNIGAELLRDPSLGAQFRVHLVKMVILTEPEGAPNITAN" \
                                               "LTSSLLSVCGWSQTINPEDDTDPGHADLVLYITRFDLELPDGNRQVRGVTQLGGACSPTWSCLITEDT" \
                                               "GFDLGVTIAHQIGHSFGLEHDGAPGSGCGPSGHVMASDGAAPRAGLAWSPCSRRQLLSLLSAGRARCV" \
                                               "WDPPRPQPGSAGHPPDAQPGLYYSANEQCRVAFGPKAVACTFAREHLDMCQALSCHTDPLDQSSCSRL" \
                                               "LVPLLDGTECGVEKWCSKGRCRSLVELTPIAAVHGRWSSWGPRSPCSRSCGGGVVTRRRQCNNPRPAF" \
                                               "GGRACVGADLQAEMCNTQACEKTQLEFMSQQCARTHWGAAVPHSQGDALCRHMCRARGDSFLDGTRCM" \
                                               "PSGPREDGTLSLCVSGSCRTFGCDGRMDSQQVWDRCQVCGGDNSTCSPRKGSFTAGRAREYVTFLTVT" \
                                               "PNLTSVYIANHRPLFTHLAVRIGGRYVVAGKMSISPNTTYPSLLEDGRVEYRVALTEDRLPRLEEIRI" \
                                               "WGPLQEDADIQVYRRYGEEYGNLTRPDITFTYFQPKP"


def test_run_smoke():
    SabdabHub(
        summary_file_path=MOCK_SABDAB_PDBS_PATH / "sabdab_mock.tsv",
        sabdab_pdb_path=MOCK_SABDAB_PDBS_PATH,
        save_path="sabdab_mock.parquet",
        re_download=False,
        n_jobs=1,
    )()
    df = pd.read_parquet("sabdab_mock.parquet")
    assert len(df) == 21
