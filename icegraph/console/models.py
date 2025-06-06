# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from tqdm import tqdm
import sys

from .objects import Spinner
from icegraph import config


class Console:
    """Class to standardize all program outputs."""

    _spinner: Spinner = None  # shared spinner

    @staticmethod
    def color(text, color) -> str:
        """Assign color to output text."""
        ansi_codes = {
            "cyan": "\u001B[36m",
            "white": "\033[37m",
            "reset": "\u001B[0m",
            "default": "\033[39m"
        }
        return ansi_codes[color] + text + ansi_codes["reset"]

    @classmethod
    def out(cls, text, control_prefix: str='', flush: bool=False, newline: bool=True, include_time: bool=True) -> None:
        """Standard output."""
        # setup output
        program_tag = f"[{cls.color(config.PROGRAM_NAME, 'cyan')}]"
        program_time = datetime.now().strftime('%X')
        delimiter = ": "

        parts = [program_tag]
        if include_time:
            parts.append(program_time)

        print(
            f"{control_prefix}{' - '.join(parts)}{delimiter}{text}",
            end="\n" if newline else ""
        )

        if flush:
            sys.stdout.flush()

    @classmethod
    def progress_bar(cls, _iter) -> iter:
        """Progress bar."""
        return tqdm(
            _iter,
            desc=f"[{cls.color(config.PROGRAM_NAME, 'cyan')}] - {datetime.now().strftime('%X')}: ",
            file=sys.stdout
        )

    @classmethod
    def spinner(cls) -> Spinner:
        """Returns the shared spinner object, creating one if it doesn't exist."""
        if Console._spinner is None:
            Console._spinner = Spinner(cls)
        return cls._spinner
