# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import xxhash
from pathlib import Path


__all__ = ["hash_directory"]

def hash_directory(input_dir: Path, config_file: Path, input_file_ext: str) -> str:
    """
    Generate a unique hash for a directory of files and a config file.

    Args:
        input_dir (Path): Path to directory containing input files.
        config_file (Path): Path to the YAML configuration file.
        input_file_ext (str): File extension of input files.

    Returns:
        str: A consistent xxHash64 hash string.
    """
    h = xxhash.xxh64()

    for file in sorted(input_dir.iterdir()):
        if file.is_file() and file.name.endswith(input_file_ext):
            h.update(file.name.encode())
            with file.open("rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)

    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    h.update(config_file.name.encode())
    with config_file.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()