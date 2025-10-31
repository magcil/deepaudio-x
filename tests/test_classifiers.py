import pytest
import torch

from deepaudiox.modules.classifier import AudioClassifierConstructor
from deepaudiox.modules.projection.projections import DivEncLayer


@pytest.mark.parametrize(
    "div_encoder_layer,sample_frequency,duration_sec",
    [
        (True, 8000, 1),
        (True, 8000, 10),
        (True, 16000, 1),
        (True, 16000, 10),
        (True, 22050, 1),
        (True, 22050, 10),
        (False, 8000, 1),
        (False, 8000, 10),
        (False, 16000, 1),
        (False, 16000, 10),
        (False, 22050, 1),
        (False, 22050, 10),
    ],
)
class TestAudioClassifierConstructor:
    """Tests for AudioClassifierConstructor logic."""

    @pytest.fixture(autouse=True)
    def setup_model(self, div_encoder_layer, sample_frequency, duration_sec):
        self.num_classes = 10
        self.duration_sec = duration_sec
        self.sample_frequency = sample_frequency
        self.samp_len = self.duration_sec * self.sample_frequency
        self.div_encoder_layer = div_encoder_layer

        projection = None
        if div_encoder_layer:
            projection = DivEncLayer(in_dim=768, out_dim=128)

        self.model = AudioClassifierConstructor(
            num_classes=self.num_classes, backbone="beats", sample_frequency=sample_frequency, projection=projection
        )

    def test_forward_waveform(self):
        x = torch.randn(4, self.samp_len)
        out = self.model(x)
        assert out.shape[0] == 4
        assert out.ndim == 2
        assert out.shape[1] == self.num_classes

    def test_get_embeddings(self):
        x = torch.randn(4, self.samp_len)
        embeddings = self.model.get_embeddings(x)
        assert embeddings.shape[0] == 4
        assert embeddings.ndim == 2
        if self.div_encoder_layer:
            assert embeddings.shape[1] == 128
        else:
            assert embeddings.shape[1] == 768
