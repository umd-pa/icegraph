# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Union, Optional

from icegraph.config import IGConfig

__all__ = ["PathResolver"]

from sympy.codegen import Attribute


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

    def resolve(self, return_dir: bool = False) -> Path:
        """
        Resolves the output path based on the provided path, origin, processing stage, and extension.

        Args:
            return_dir (bool): If True, returns a directory path. If False, returns a full file path.

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




