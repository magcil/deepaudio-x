import librosa
import numpy as np
import torchaudio


def load_audio(
    file_path: str, 
    sample_rate: int = 16000, 
    start_sample: int = 0, 
    end_sample: int | None = None
) -> np.ndarray:
    """Load an audio file.

    Args:
        file_path (str): The path to the audio file
        sample_rate (int): The audio sampling rate. Defaults to 16000.
        start_sample (int): The starting sample point, when cropping. Defaults to 0.
        end_sample (int OR None): The ending sample point, when cropping. Defaults to None.

    Returns:
        torch.Tensor: The Torch tensor with the audio data

    """
    waveform, sr = librosa.load(file_path, sr=sample_rate)

    if end_sample is None:
        end_sample = len(waveform)
    waveform = waveform[start_sample:end_sample]

    return waveform

def get_audio_num_samples(file_path: str) -> int:
    """Return the number of sample points of an audio file, without loading it.

    Args:
        file_path (str): The path to the audio file
    Returns:
        int: The number of sample points

    """
    info = torchaudio.info(file_path)
    return info.num_frames
