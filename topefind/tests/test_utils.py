import numpy as np

from topefind.utils import (
    natsort,
    pad_to_imgt,
    get_antibody_numbering,
    get_antibody_regions
)

AB_IMGT = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
    '22', '23', '24', '25', '26', '27', '28', '29', '30', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44',
    '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '62', '63', '64', '65',
    '66', '67', '68', '69', '70', '71', '72', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
    '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
    '104', '105', '106', '107', '108', '109', '110', '111', '111A', '112A', '112', '113', '114', '115', '116', '117',
    '118', '119', '120', '121', '122', '123', '124', '125', '126'
]

AB_SEQ = [
    aa for aa in
    "QVQLQQSGAELVKPGASVRMSCKASGYTFTNYNMYWVKQSPGQGLEWIGIFYPGNGDTSY"
    "NQKFKDKATLTADKSSNTAYMQLSSLTSEDSAVYYCARSGGSYRYDGGFDYWGQGTTVTV"
]

FULL_PARA_LABELS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0
]

AB_IMGT_PADDED = np.array([
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '11', '12', '13',
    '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
    '25', '26', '27', '28', '29', '30', '-', '-', '-', '-', '35', '36',
    '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
    '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
    '59', '-', '-', '62', '63', '64', '65', '66', '67', '68', '69',
    '70', '71', '72', '-', '74', '75', '76', '77', '78', '79', '80',
    '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91',
    '92', '93', '94', '95', '96', '97', '98', '99', '100', '101',
    '102', '103', '104', '105', '106', '107', '108', '109', '110',
    '111', '111A', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-',
    '-', '-', '-', '-', '-', '-', '112A', '112', '113', '114', '115',
    '116', '117', '118', '119', '120', '121', '122', '123', '124',
    '125', '126', '-'
])

AB_SEQ_PADDED = np.array([
    'Q', 'V', 'Q', 'L', 'Q', 'Q', 'S', 'G', 'A', '-', 'E', 'L', 'V',
    'K', 'P', 'G', 'A', 'S', 'V', 'R', 'M', 'S', 'C', 'K', 'A', 'S',
    'G', 'Y', 'T', 'F', '-', '-', '-', '-', 'T', 'N', 'Y', 'N', 'M',
    'Y', 'W', 'V', 'K', 'Q', 'S', 'P', 'G', 'Q', 'G', 'L', 'E', 'W',
    'I', 'G', 'I', 'F', 'Y', 'P', 'G', '-', '-', 'N', 'G', 'D', 'T',
    'S', 'Y', 'N', 'Q', 'K', 'F', 'K', '-', 'D', 'K', 'A', 'T', 'L',
    'T', 'A', 'D', 'K', 'S', 'S', 'N', 'T', 'A', 'Y', 'M', 'Q', 'L',
    'S', 'S', 'L', 'T', 'S', 'E', 'D', 'S', 'A', 'V', 'Y', 'Y', 'C',
    'A', 'R', 'S', 'G', 'G', 'S', 'Y', 'R', '-', '-', '-', '-', '-',
    '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'Y', 'D',
    'G', 'G', 'F', 'D', 'Y', 'W', 'G', 'Q', 'G', 'T', 'T', 'V', 'T',
    'V', '-'
])

FULL_PARA_LABELS_PADDED = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -100, -100, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
    0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100
])


def test_natsort():
    test_list = ["100A", "1A", "2B", "10", "100B", "100", "100C", "5"]
    expected = ["1A", "2B", "5", "10", "100", "100A", "100B", "100C"]
    assert all([ex == sor for ex, sor in zip(expected, natsort(test_list))])


def test_pad_to_imgt_seq():
    padded_seq = pad_to_imgt(
        arr_to_pad=np.array(AB_SEQ),
        antibody_imgt=np.array(AB_IMGT),
        max_imgt_pos="128"
    )
    assert np.all(padded_seq == AB_SEQ_PADDED)


def test_pad_to_imgt_labels():
    padded_labels = pad_to_imgt(
        arr_to_pad=np.array(FULL_PARA_LABELS),
        antibody_imgt=np.array(AB_IMGT),
        max_imgt_pos="128"
    )
    assert np.all(padded_labels == FULL_PARA_LABELS_PADDED)


def test_pad_to_imgt_imgt():
    padded_imgt = pad_to_imgt(
        arr_to_pad=np.array(AB_IMGT),
        antibody_imgt=np.array(AB_IMGT),
        max_imgt_pos="128"
    )
    assert np.all(padded_imgt == AB_IMGT_PADDED)


def test_get_antibody_numbering():
    expected = np.array(AB_IMGT)
    test_seq = "".join(AB_SEQ)
    assert np.all(expected == get_antibody_numbering(sequence=test_seq))


def test_get_antibody_regions():
    expected_cdrs = [
        # cdr1
        ['A', 'S', 'G', 'Y', 'T', 'F', 'T', 'N', 'Y', 'N', 'M', 'Y', 'W'],
        # cdr2
        ['I', 'F', 'Y', 'P', 'G', 'N', 'G', 'D', 'T', 'S'],
        # cdr3
        ['A', 'R', 'S', 'G', 'G', 'S', 'Y', 'R', 'Y', 'D', 'G', 'G', 'F', 'D', 'Y', 'W', 'G']
    ]
    expected_fmks = [
        # fmk1
        ['Q', 'V', 'Q', 'L', 'Q', 'Q', 'S', 'G', 'A', 'E', 'L', 'V', 'K', 'P',
         'G', 'A', 'S', 'V', 'R', 'M', 'S', 'C', 'K'],
        # fmk2
        ['V', 'K', 'Q', 'S', 'P', 'G', 'Q', 'G', 'L', 'E', 'W', 'I', 'G'],
        # fmk3
        ['Y', 'N', 'Q', 'K', 'F', 'K', 'D', 'K', 'A', 'T', 'L', 'T', 'A', 'D', 'K', 'S', 'S', 'N', 'T',
         'A', 'Y', 'M', 'Q', 'L', 'S', 'S', 'L', 'T', 'S', 'E', 'D', 'S', 'A', 'V', 'Y', 'Y', 'C'],
        # fmk4
        ['Q', 'G', 'T', 'T', 'V', 'T', 'V']
    ]

    cdrs, fmks = get_antibody_regions("".join(AB_SEQ))
    for cdr, ecdr in zip(cdrs, expected_cdrs):
        assert np.all(np.array(cdr) == np.array(ecdr))
    for fmk, efmk in zip(fmks, expected_fmks):
        assert np.all(np.array(fmk) == np.array(efmk))
