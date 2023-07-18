from pathlib import Path

import numpy as np
import pytest

from topefind.predictors import (
    EndToEndPredictorName,
    Parapred,
    Paragraph,
)

TEST_SEQ = "QVQLQESGPGLVRPSQTLSLTCTVSGFSLTGYGVNWVRQPPGRGLEWIGMIWGDGNTD" \
           "YNSALKSRVTMLKDTSKNQFSLRLSSVTAADTAVYYCARERDYRLDYWGQGSLVTVSS"

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

