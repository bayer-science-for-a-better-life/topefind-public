from topefind.embedders import (
    ESMEmbedder,
    RITAEmbedder,
    PhysicalPropertiesNoPosEmbedder,
    PhysicalPropertiesPosEmbedder,
    PhysicalPropertiesPosContextEmbedder,
    IMGTPosEmbedder,
    EmbedderName,
)

import pytest

TEST_SEQ = "QVQLQESGPGLVRPSQTLSLTCTVSGFSLTGYGVNWVRQPPGRGLEWIGMIWGDGNTD" \
           "YNSALKSRVTMLKDTSKNQFSLRLSSVTAADTAVYYCARERDYRLDYWGQGSLVTVSS"
MODELS = [
    (ESMEmbedder, EmbedderName.esm2_8m),
    (RITAEmbedder, EmbedderName.rita_s),
    (PhysicalPropertiesNoPosEmbedder, EmbedderName.aa),
    (IMGTPosEmbedder, EmbedderName.imgt),
    (PhysicalPropertiesPosEmbedder, EmbedderName.imgt_aa),
    (PhysicalPropertiesPosContextEmbedder, EmbedderName.imgt_aa_ctx_3),
]


# This would be a nice way to do it...
# However, running 3B models on CI is not really ideal...
# So, currently testing only smaller model.

@pytest.fixture(params=MODELS, scope="class")
def embedder_init(request):
    request.cls.embedder = request.param[0](name=request.param[1])
    yield


@pytest.mark.usefixtures("embedder_init")
class TestBase:
    pass


class TestEmbedder(TestBase):
    def test_smoke(self):
        self.embedder.embed(TEST_SEQ)

    def test_lengths(self):
        embs = self.embedder.embed(TEST_SEQ)
        assert len(TEST_SEQ) == len(embs[0])

    def test_smoke_multiple(self):
        self.embedder.embed([TEST_SEQ] * 3)
