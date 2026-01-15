import pytest
import torch

from deepaudiox.modules.audio_classifier_constructor import AudioClassifierConstructor


@pytest.mark.parametrize(
    "sample_rate,duration_sec",
    [
        (8000, 3),
        (16000, 1),
        (22050, 5),
        (8000, 10),
        (16000, 1),
        (22050, 1),
    ],
)
class TestAudioClassifierConstructor:
    """Tests for AudioClassifierConstructor logic."""

    @pytest.fixture(autouse=True)
    def setup_model(self, sample_rate, duration_sec):
        self.num_classes = 10
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.samp_len = self.duration_sec * self.sample_rate

        self.model = AudioClassifierConstructor(num_classes=self.num_classes, backbone="beats", sample_rate=sample_rate)

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
        assert embeddings.ndim == 3
        assert embeddings.shape[-1] == 768