from pathlib import Path

import numpy as np
import pytest

from topefind.predictors import (
    EndToEndPredictorName,
    Parapred,
    Paragraph,
    AAFrequency,
    PosFrequency,
    AAPosFrequency,
)

TEST_SEQ = "QVQLQESGPGLVRPSQTLSLTCTVSGFSLTGYGVNWVRQPPGRGLEWIGMIWGDGNTD" \
           "YNSALKSRVTMLKDTSKNQFSLRLSSVTAADTAVYYCARERDYRLDYWGQGSLVTVSS"

TEST_LABELS = [False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, True, True, True, False, False, False, False,
               False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, False, True, True, True, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False, False, False, False, False,
               False, False, True, True, True, True, False, False, False, False, False, False,
               False, False, False, False, False, False, False, False]

TEST_IMGT = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '35',
             '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
             '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '63', '64', '65', '66',
             '67', '68', '69', '70', '71', '72', '74', '75', '76', '77', '78', '79', '80', '81',
             '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95',
             '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108',
             '109', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123',
             '124', '125', '126', '127', '128']

TEST_PDB_DICT = {"H_id": ["B"], "L_id": ["A"], "pdb_code": ["1bvk"], "focus": "B"}

ZEROS_SEQ = np.zeros(len(TEST_SEQ))
FILE_PATH = Path(__file__)


class TestParapred:
    @classmethod
    def setup_class(cls):
        cls.model = Parapred(EndToEndPredictorName.parapred)

    def test_smoke(self):
        self.model.predict(TEST_SEQ)

    def test_lengths(self):
        probs = self.model.predict(TEST_SEQ)
        assert len(TEST_SEQ) == len(probs)

    def test_prob_outputs(self):
        # We expect to have probabilities
        probs = self.model.predict(TEST_SEQ)
        assert np.allclose(ZEROS_SEQ + 0.5, probs, atol=0.5)

    def test_all_diff(self):
        # Expecting to be all different, at least from the first one
        probs = self.model.predict(TEST_SEQ)
        assert not np.allclose(probs, probs[0])

    def test_not_zeros(self):
        probs = self.model.predict(TEST_SEQ)
        assert not np.allclose(probs, ZEROS_SEQ)

    def test_null_input(self):
        with pytest.raises(TypeError):
            self.model.predict("")

    def test_smoke_multiple(self):
        self.model.predict_multiple([TEST_SEQ] * 3)


class TestParagraph:
    @classmethod
    def setup_class(cls):
        cls.model = Paragraph(
            EndToEndPredictorName.paragraph_unpaired,
            FILE_PATH.parent / "mock_data" / "imgt"
        )

    def test_smoke(self):
        self.model.predict(TEST_PDB_DICT)

    def test_lengths(self):
        probs = self.model.predict(TEST_PDB_DICT)
        assert len(TEST_SEQ) == len(probs)

    def test_prob_outputs(self):
        # We expect to have probabilities
        probs = self.model.predict(TEST_PDB_DICT)
        assert np.allclose(ZEROS_SEQ + 0.5, probs, atol=0.5)

    def test_all_diff(self):
        # Expecting to be all different, at least from the first one
        probs = self.model.predict(TEST_PDB_DICT)
        assert not np.allclose(probs, probs[0])

    def test_not_zeros(self):
        probs = self.model.predict(TEST_PDB_DICT)
        assert not np.allclose(probs, ZEROS_SEQ)

    def test_smoke_multiple(self):
        self.model.predict_multiple([TEST_PDB_DICT] * 3)


class TestAAFrequency:
    @classmethod
    def setup_class(cls):
        cls.model = AAFrequency(EndToEndPredictorName.aa_freq)

    def test_smoke(self):
        preds = self.model.predict(TEST_SEQ)
        assert np.sum(preds) == 0

        self.model.train([TEST_SEQ], [TEST_LABELS])
        preds = self.model.predict(TEST_SEQ)
        assert np.sum(preds) != 0


class TestPosFrequency:
    @classmethod
    def setup_class(cls):
        cls.model = PosFrequency(EndToEndPredictorName.pos_freq)

    def test_smoke(self):
        preds = self.model.predict(TEST_SEQ)
        assert np.sum(preds) == 0

        self.model.train([TEST_SEQ], [TEST_LABELS])
        preds = self.model.predict(TEST_SEQ)
        assert np.sum(preds) != 0


class TestAAPosFrequency:
    @classmethod
    def setup_class(cls):
        cls.model = AAPosFrequency(EndToEndPredictorName.aa_pos_freq)

    def test_smoke(self):
        preds = self.model.predict(TEST_SEQ)
        assert np.sum(preds) == 0

        self.model.train([TEST_SEQ], [TEST_LABELS])
        preds = self.model.predict(TEST_SEQ)
        print(preds)
        assert np.sum(preds) != 0
