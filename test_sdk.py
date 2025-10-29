import torch.nn as nn
import torch.optim as optim

from src.deepaudiox.datasets.audio_classification_dataset import AudioClassificationDataset
from src.deepaudiox.loops.trainer import Trainer
from src.deepaudiox.models.wav2vec_classifier import Wav2VecClassifier
from src.deepaudiox.utils.training_utils import get_class_mapping


def main():
    # Initialize model
    model = Wav2VecClassifier()

    # Dataset
    root_dir = "..."
    class_mapping = get_class_mapping(root_dir)
    train_dset = AudioClassificationDataset(
        root_dir=root_dir,
        sample_rate=16000,
        class_mapping=class_mapping
    )

    # Define optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
    loss_function = nn.CrossEntropyLoss()

    # Initialize trainer
    trainer = Trainer(
        train_dset=train_dset,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        lr_scheduler=scheduler,
        train_ratio=0.8,
        epochs=100,
        learning_rate=1e-3,
        patience=5,
        num_workers=4,
        batch_size=16
    )

    trainer.train()


if __name__ == '__main__':
    main()
