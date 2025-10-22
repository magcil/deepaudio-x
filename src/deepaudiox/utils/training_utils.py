from pathlib import Path


def get_class_mapping(root_dir: str):
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