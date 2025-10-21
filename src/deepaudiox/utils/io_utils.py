import csv
import json
from pathlib import Path
from typing import Any

import yaml


def load_csv(file_path: str, has_header: bool = True) -> Any:
    """Load the content of a CSV file.

    Args:
        file_path (str): The path to the CSV file
        has_header (bool): Determines if CSV file includes a header row

    Returns:
        Any: The list of CSV rows, or None (in case of error)

    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        with file_path.open(mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f) if has_header else csv.reader(f)
            return list(reader)
    except OSError as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_json(file_path: str) -> Any:
    """Load the content of a JSON file.

    Args:
        file_path (str): The path to the JSON file

    Returns:
        Any: The content of the JSON file, or None (in case of error)

    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return None


def load_yaml(file_path: str) -> Any:
    """Load the content of a YAML file.

    Args:
        file_path (str): The path to the YAML file

    Returns:
        Any: The content of the YAML file, or None (in case of error)

    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"YAML parsing error in {file_path}: {e}")
        return None
    except OSError as e:
        print(f"Error reading {file_path}: {e}")
        return None