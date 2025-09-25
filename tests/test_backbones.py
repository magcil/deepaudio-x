import pytest
import torch

from deepaudiox.backbones.beats.beatswrapper import BEATsBackbone


@pytest.mark.parametrize("div_encoder_layer,sample_frequency,duration_sec", [
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
])
class TestBEATsBackbone:
    """Tests for BEATsBackbone model."""
    @pytest.fixture(autouse=True)
    def setup_model(self, div_encoder_layer, sample_frequency, duration_sec):
        self.model = BEATsBackbone(
            div_encoder_layer=div_encoder_layer,
            sample_frequency=sample_frequency,
        )
        self.duration_sec = duration_sec
        self.sample_frequency = sample_frequency
        self.samp_len = self.duration_sec * self.sample_frequency

    def test_forward_waveform(self):
        x = torch.randn(4, self.samp_len)
        out = self.model(x)
        assert out.shape[0] == 4    
        assert out.ndim == 2
        if self.model.div_encoder_layer:
            assert out.shape[1] == 128
        else:
            assert out.shape[1] == 768

    def test_freeze_encoder_weights(self):
        self.model.freeze_encoder_weights()
        assert all(not p.requires_grad for p in self.model.encoder.parameters())

