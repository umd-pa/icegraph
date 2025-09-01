# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
import sys
import threading
from typing import Optional, Sequence, Union, List, ClassVar
from pathlib import Path
from wcwidth import wcswidth
import re

from rich.progress import track
from rich.console import Console as RichConsole

from .objects import Spinner
from icegraph.config import IGConfig

__all__ = ["Console"]


class Console:
    """
    Class to standardize all console outputs across the application.

    Provides unified formatting for standard output, progress bars, and spinners.
    """

    _spinner: Optional[Spinner] = None  # Shared spinner instance
    _is_internal_write = threading.local()  # Thread-local flag to detect Console-originated output
    _is_internal_write.value = False

    ANSI_RE: ClassVar[re.Pattern] = re.compile(r"\x1b\[[0-9;]*m")

    @staticmethod
    def color(text: str, color: str) -> str:
        """
        Apply ANSI color codes to text for terminal output.

        Args:
            text (str): The text to color.
            color (str): Color name.

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
        """
        Map a numeric severity level to its corresponding tag string.

        Args:
            severity (int): Numeric severity level
                - 0: INFO
                - 1: IMPT
                - 2: WARN
                - 3: CRIT

        Returns:
            str: The severity tag, with appropriate ANSI coloring for IMPT/WARN/CRIT.
        """
        mapping = {
            0: "INFO",
            1: cls.color("IMPT", "green"),
            2: cls.color("WARN", "orange"),
            3: cls.color("CRIT", "red")
        }
        return mapping[severity]

    @classmethod
    def _apply_severity(cls, text: str, severity: int) -> str:
        """
        Colorize a text string based on its severity level.

        Args:
            text (str): The input text to color.
            severity (int): Numeric severity level
                - 0: default (no color change)
                - 1: green
                - 2: orange
                - 3: red

        Returns:
            str: The input text wrapped in the ANSI codes for the given severity.
        """
        mapping = {
            0: "default",
            1: "green",
            2: "orange",
            3: "red"
        }
        return cls.color(text, color=mapping[severity])

    @classmethod
    def _visible_len(cls, string: str) -> int:
        # strip ANSI; then measure display width
        return wcswidth(cls.ANSI_RE.sub("", string))

    @classmethod
    def multi_out(
        cls,
        lines: List[str],
        severity: int = 0
    ):
        """
        Print multiple lines out to console at once.

        Args:
            lines (List[str]): The list of messages to print.
            severity (int): Severity level, integer from 0 to 3 representing INFO, IMPT, WARN, and CRIT. Defaults to 0.
        """
        _first_line = False
        for text in lines:
            cls.out(text, severity, replace_tag_with_indent=_first_line)
            if not _first_line:
                _first_line = True

    @classmethod
    def out(
        cls,
        text: str,
        severity: int = 0,
        control_prefix: str = '',
        flush: bool = False,
        newline: bool = True,
        include_info: bool = True,
        replace_tag_with_indent: bool = False
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
            replace_tag_with_indent (bool): Replace the info and tag with an indent of equal size.
        """
        cls._is_internal_write.value = True  # Mark output as Console-generated
        try:
            program_tag = f"[{cls.color(IGConfig.PROGRAM_NAME, 'cyan')}]"
            program_time = datetime.now().strftime('%X')
            severity_tag = cls._severity_tag(severity)
            delimiter = ": "
            text = cls._apply_severity(text, severity)

            parts = [program_tag]
            if include_info:
                parts.append(program_time)
                parts.append(severity_tag)

            if not replace_tag_with_indent:
                print(
                    f"{control_prefix}{' '.join(parts)}{delimiter}{text}",
                    end="\n" if newline else ""
                )
            else:
                indent_size = cls._visible_len(' '.join(parts) + delimiter)
                print(
                    f"{control_prefix}{' ' * indent_size}{text}",
                    end="\n" if newline else ""
                )

            if flush:
                sys.stdout.flush()
        finally:
            cls._is_internal_write.value = False  # Reset flag

    @classmethod
    def progress_bar(cls, _iter, **kwargs) -> iter:
        """
        Create a standardized progress bar using `rich`.

        Args:
            _iter (iterable): The iterable to wrap in a progress bar.

        Returns:
            iterator: The wrapped iterable with progress bar display.
        """
        total = kwargs.pop("total", None)

        # If total isn't given, track() will try len(_iter) if available
        if total is None and hasattr(_iter, "__len__"):
            try:
                total = len(_iter)  # type: ignore
            except Exception:
                total = None

        # description
        program_tag = f"[{cls.color(IGConfig.PROGRAM_NAME, 'cyan')}]"
        program_time = datetime.now().strftime('%X')
        severity_tag = cls._severity_tag(0)
        delimiter = ": "

        parts = [program_tag, program_time, severity_tag]

        console = RichConsole(file=sys.stdout, force_terminal=True, force_interactive=True)

        return track(
            _iter,
            total=total,
            description=" ".join(parts) + delimiter,
            console=console,
            update_period=0.1
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

    @classmethod
    def newline(cls) -> None:
        """
        Prints an empty line.
        """
        print("")

    @classmethod
    def banner(cls, stage: str) -> None:
        """
        Prints a banner for the current processing stage.
        """
        cls.newline()
        print("=" * 15 + " " + "ICEGRAPH --- " + stage + " " + "=" * 15)

    @classmethod
    def source_repr(cls, source: Union[str, Path, Sequence[Union[str, Path]]]) -> str:
        """Get a CLI friendly string representation of a data source."""
        printout: str
        if isinstance(source, Sequence) and not isinstance(source, (str, Path)):
            printout = f"[{source[0]!s}, ...]"
        else:
            printout = source
        return printout
