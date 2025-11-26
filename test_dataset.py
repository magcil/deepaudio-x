from deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from deepaudiox.utils.training_utils import get_class_mapping

dataset = AudioClassificationDataset(
    root_dir = "/data/ESC-50-master/Sorted", 
    sample_rate =  16000,
    class_mapping = get_class_mapping("/data/ESC-50-master/Sorted"),
    instance_metadata = [
        {
            "file_name": "1-11687-A-47.wav",
            "class_name": "airplane"
        },
        {
            "file_name": "1-25777-A-48.wav",
            "class_name": "fireworks"
        }
    ],
    segment_duration =  1
)


for item in dataset:
    print(item)