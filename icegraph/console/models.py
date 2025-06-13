# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from tqdm import tqdm
import sys
import threading
from typing import Optional

from .objects import Spinner
from icegraph.config import IGConfig


__all__ = ["Console"]

class Console:
    """
    Class to standardize all console outputs across the application.

    Provides unified formatting for standard output, progress bars, and spinners.
    """

    _spinner: Optional['Spinner'] = None  # Shared spinner instance
    _is_internal_write = threading.local()  # Thread-local flag to detect Console-originated output
    _is_internal_write.value = False

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
            "default": "\033[39m",
            "red": "\033[31m",
            "yellow": "\033[33m",
            "orange": "\033[38;2;255;165;0m",
            "green": "\033[32m"
        }
        return ansi_codes[color] + text + ansi_codes["reset"]

    @classmethod
    def _severity_tag(cls, severity: int) -> str:
        mapping = {
            0: "INFO",
            1: cls.color("IMPT", "green"),
            2: cls.color("WARN", "orange"),
            3: cls.color("CRIT", "red")
        }
        return mapping[severity]

    @classmethod
    def out(
        cls,
        text: str,
        severity: int = 0,
        control_prefix: str = '',
        flush: bool = False,
        newline: bool = True,
        include_info: bool = True
    ) -> None:
        """
        Print standardized program output to stdout.

        Args:
            text (str): The message to print.
            severity (int): Severity level, integer from 0 to 3 representing INFO, IMPT, WARN, and CRIT. Defaults to 0.
            control_prefix (str): Optional prefix (e.g., indentation or control characters).
            flush (bool): Whether to flush stdout immediately.
            newline (bool): Whether to append a newline character.
            include_info (bool): Whether to include timestamp/severity in the output.
        """
        cls._is_internal_write.value = True  # Mark output as Console-generated
        try:
            program_tag = f"[{cls.color(IGConfig.PROGRAM_NAME, 'cyan')}]"
            program_time = datetime.now().strftime('%X')
            severity_tag = cls._severity_tag(severity)
            delimiter = ": "

            parts = [program_tag]
            if include_info:
                parts.append(program_time)
                parts.append(severity_tag)

            print(
                f"{control_prefix}{' '.join(parts)}{delimiter}{text}",
                end="\n" if newline else ""
            )

            if flush:
                sys.stdout.flush()
        finally:
            cls._is_internal_write.value = False  # Reset flag

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
            desc=f"[{cls.color(IGConfig.PROGRAM_NAME, 'cyan')}] - {datetime.now().strftime('%X')}: ",
            file=sys.stdout
        )

    @classmethod
    def spinner(cls) -> 'Spinner':
        """
        Access the shared Spinner object.

        Returns:
            Spinner: The shared spinner instance.
        """
        if Console._spinner is None:
            Console._spinner = Spinner(cls)
        return cls._spinner