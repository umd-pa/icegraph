# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Union, Optional, Sequence, List
import os

from icegraph.config import IGConfig

__all__ = ["PathResolver", "PathValidator"]


class PathResolver:
    """
    A utility for resolving file or directory paths used in data processing pipelines.
    """

    def __init__(self, path: Optional[Union[str, Path]], origin: Optional[Union[str, Path]], extension: Optional[str], stage: str) -> None:
        self.path = Path(path) if path is not None else None
        self.origin = Path(origin) if origin is not None else None
        if extension:
            self.extension = extension if extension.startswith('.') else f'.{extension}'
        self.stage = stage

        # grab global config
        config = IGConfig.get()
        self.default_dir = Path(config.user_config.io.default_dir)

    def resolve(self, return_dir: bool = False, prefix: Optional[str] = None) -> Path:
        """
        Resolves the output path based on the provided path, origin, processing stage, and extension.

        Args:
            return_dir (bool): If True, returns a directory path. If False, returns a full file path.
            prefix (Optional[str]): Optionally specify a desired file name prefix.

        Returns:
            Path: The resolved path.
        """
        path = self.path

        if return_dir:
            if path is None:
                resolve_path = self.default_dir / self.stage
            else:
                resolve_path = path if self._is_dirlike(path) else path.parent
        else:
            if not hasattr(self, "extension"):
                raise AttributeError("PathResolver attribute 'extension' cannot be None if resolving a file path.")

            if prefix is not None:
                inferred_name = prefix + self.extension
            else:
                if self.origin is not None:
                    inferred_name = self.origin.with_suffix(self.extension).name
                else:
                    inferred_name = self.stage + "_outfile" + self.extension

            if path is None:
                resolve_path = self.default_dir / self.stage / inferred_name
            else:
                resolve_path = path / inferred_name if self._is_dirlike(path) else path.with_suffix(self.extension)

        self._make_dirs(resolve_path)
        return resolve_path

    def _make_dirs(self, path: Union[str, Path]) -> None:
        """
        Ensures that the directory corresponding to the given path exists.

        Args:
            path (Union[str, Path]): The file or directory path whose parent or self should exist.
        """
        path = Path(path)
        target = path if self._is_dirlike(path) else path.parent
        target.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_dirlike(path: Path) -> bool:
        """
        Determines whether a path should be treated as a directory based on its suffix.

        A path is considered 'directory-like' if it has no suffix (i.e., no file extension).

        Returns:
            bool: True if the path has no suffix, indicating it is directory-like; False otherwise.
        """
        return Path(path).suffix == ""

    @staticmethod
    def normalize_sources(
            source: Union[str, Path, Sequence[Union[str, Path]]],
            extension: str,
            use_str: bool = False
    ) -> Union[List[str], List[Path]]:
        extension = extension if extension.startswith(".") else "." + extension
        want_suffixes = extension.lower().split(".")
        want_suffixes = ["." + s for s in want_suffixes if s]

        # Fast path for single extension
        single_ext = (len(want_suffixes) == 1)
        single_ext_lc = want_suffixes[0] if single_ext else None

        def has_wanted_suffixes(p: Path) -> bool:
            if single_ext:
                return p.name.lower().endswith(single_ext_lc)
            # multi extension case: suffix chain check
            fs = [s.lower() for s in p.suffixes]
            n = len(want_suffixes)
            return len(fs) >= n and fs[-n:] == want_suffixes

        def to_paths(obj: Union[str, Path]) -> List[Path]:
            p = Path(obj).expanduser()
            PathValidator.is_valid_path(p)
            if p.is_dir():
                # deterministic order
                entries = sorted(p.iterdir(), key=lambda y: y.name)
                return [x.resolve() for x in entries if x.is_file() and has_wanted_suffixes(x)]
            q = p.resolve()
            return [q] if q.is_file() and has_wanted_suffixes(q) else []

        # collect
        files: List[Path] = []
        if isinstance(source, (str, Path)):
            files = to_paths(source)
        else:
            for s in source:
                files.extend(to_paths(s))

        # dedupe while preserving order
        seen = set()
        out: Union[List[str], List[Path]] = []
        for f in files:
            if f not in seen:
                out.append(str(f) if use_str else f)
                seen.add(f)

        return out


class PathValidator:

    @classmethod
    def is_valid_file(cls, path: Union[str, Path]) -> None:
        """
        Check that the given path exists, is a file, and is readable.
        Raises informative exceptions if any check fails.

        Args:
            path (str or Path): The file path to check.

        Raises:
            FileNotFoundError: If the path does not exist.
            IsADirectoryError: If the path is a directory, not a file.
            PermissionError: If the file is not readable.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Expected a file but found a directory or special file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"File is not readable: {path}")

    @classmethod
    def is_valid_dir(cls, path: Union[str, Path]) -> None:
        """
        Check that the given path exists, is a directory, and is readable.
        Raises informative exceptions if any check fails.

        Args:
            path (str or Path): The file path to check.

        Raises:
            FileNotFoundError: If the path does not exist.
            NotADirectoryError: If the path is a file, not a directory.
            PermissionError: If the file is not readable.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Expected a directory but found a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Directory is not readable: {path}")

    @classmethod
    def is_valid_path(cls, path: Union[str, Path]) -> None:
        """
        Check that the given path exists and is readable.
        Raises informative exceptions if any check fails.

        Args:
            path (str or Path): The path to check.

        Raises:
            FileNotFoundError: If the path does not exist.
            PermissionError: If the file is not readable.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Resource does not exist: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Resource is not readable: {path}")

