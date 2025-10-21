from pathlib import Path

from sklearn.model_selection import train_test_split


def get_class_mapping_from_folder(root_dir: str):
    """Load the class mapping given a folder of class sub-folders.

    Args:
        root_dir (str): The path to root folder

    Returns:
        Dict: The class mapping dictionary

    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"The path '{root_dir}' is not a directory")

    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    class_mapping = {d.name: idx for idx, d in enumerate(sorted(subdirs))}

    return class_mapping

def split_folder(root_dir: str, ratio: float, seed: int):
    """Split a dataset folder into training and validation sets.

        Scans the root directory for subfolders (each representing a class),
        collects all `.wav` and `.mp3` files, and performs a stratified split
        into training and validation subsets.

        Args:
            root_dir (str): Path to the root dataset folder. Each subfolder is treated as a class.
            ratio (float): Proportion of data to use for validation.
            seed (int): Random seed for reproducibility of the split.

        Returns:
            tuple:
                train_paths (list[str]): List of file paths for training samples.
                train_classes (list[str]): Corresponding class labels for training samples.
                val_paths (list[str]): List of file paths for validation samples.
                val_classes (list[str]): Corresponding class labels for validation samples.

    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"The path '{root_dir}' is not a directory")
    
    instance_paths = []
    instance_classes = []

    # Collect all .wav and .mp3 files along with their class names
    for child_directory in root_path.iterdir():
        if child_directory.is_dir():
            for audio_file in child_directory.rglob("*.wav"):
                instance_paths.append(str(audio_file))
                instance_classes.append(child_directory.name)
            for audio_file in child_directory.rglob("*.mp3"):
                instance_paths.append(str(audio_file))
                instance_classes.append(child_directory.name)

    if len(instance_classes) == 0:
        raise ValueError("No labels found in this folder for split.")

    # Split into train and validation sets
    train_paths, val_paths, train_classes, val_classes = train_test_split(
        instance_paths,
        instance_classes,
        stratify=instance_classes,
        test_size=ratio,
        random_state=seed
    )

    return train_paths, train_classes, val_paths, val_classes