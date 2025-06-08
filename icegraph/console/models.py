# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from tqdm import tqdm
import sys

from .objects import Spinner
from icegraph.config import Config


__all__ = ["Console"]

class Console:
    """
    Class to standardize all console outputs across the application.

    Provides unified formatting for standard output, progress bars, and spinners.
    """

    _spinner: Spinner | None = None  # Shared spinner instance

    @staticmethod
    def color(text: str, color: str) -> str:
        """
        Apply ANSI color codes to text for terminal output.

        Args:
            text (str): The text to color.
            color (str): Color name ('cyan', 'white', 'reset', or 'default').

        Returns:
            str: The colored text.
        """
        ansi_codes = {
            "cyan": "\u001B[36m",
            "white": "\033[37m",
            "reset": "\u001B[0m",
            "default": "\033[39m"
        }
        return ansi_codes[color] + text + ansi_codes["reset"]

    @classmethod
    def out(
        cls,
        text: str,
        control_prefix: str = '',
        flush: bool = False,
        newline: bool = True,
        include_time: bool = True
    ) -> None:
        """
        Print standardized program output to stdout.

        Args:
            text (str): The message to print.
            control_prefix (str): Optional prefix (e.g., indentation or control characters).
            flush (bool): Whether to flush stdout immediately.
            newline (bool): Whether to append a newline character.
            include_time (bool): Whether to include a timestamp in the output.
        """
        program_tag = f"[{cls.color(Config.PROGRAM_NAME, 'cyan')}]"
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
        """
        Create a standardized progress bar using `tqdm`.

        Args:
            _iter (iterable): The iterable to wrap in a progress bar.

        Returns:
            iterator: The wrapped iterable with progress bar display.
        """
        return tqdm(
            _iter,
            desc=f"[{cls.color(Config.PROGRAM_NAME, 'cyan')}] - {datetime.now().strftime('%X')}: ",
            file=sys.stdout
        )

    @classmethod
    def spinner(cls) -> Spinner:
        """
        Access the shared Spinner object.

        Returns:
            Spinner: The shared spinner instance.
        """
        if Console._spinner is None:
            Console._spinner = Spinner(cls)
        return cls._spinner