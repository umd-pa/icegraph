# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from tqdm import tqdm


class Console:
    """Class to standardize all program outputs."""

    @staticmethod
    def color(text, color) -> str:
        ansi_codes = {
            "cyan": "\u001B[36m",
            "reset": "\u001B[0m"
        }
        return ansi_codes[color] + text + ansi_codes["reset"]

    @classmethod
    def out(cls, text) -> None:
        """Standard output."""
        print(f"[{cls.color('icegraph', 'cyan')}] - {datetime.now().strftime('%X')}: {text}")

    @classmethod
    def prog(cls, _iter) -> iter:
        """Progress bar."""
        return tqdm(_iter)