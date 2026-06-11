# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Iterable, Sequence, Self
from pathlib import Path
from dataclasses import dataclass
from itertools import chain


SourceType: TypeAlias = str | Path | Sequence[str | Path]
SourceStruct: TypeAlias = str | list[str]

@dataclass(frozen=True)
class Source:
    """
    Represents a user-supplied source of files.

    A Source may be:
      - a path to a file
      - a path to a directory
      - a sequence of paths to files or directories
      - a combination of the above

    ``resolve()`` yields matching files lazily.
    """
    source: SourceType | Source

    def __post_init__(self) -> None:
        if isinstance(self.source, Source):
            object.__setattr__(self, "source", self.source.source)

    @staticmethod
    def _expand_path(path: Path, extension: str, recursive: bool) -> Iterable[Path]:
        # expand user home dir
        path = path.expanduser()

        if path.is_file():
            # if source is one file, return it as a list (if extension matches)
            return [path] if path.name.lower().endswith(extension) else []

        elif path.is_dir():
            # if leads to a dir, glob
            pattern = f"*{extension}"
            return path.rglob(pattern) if recursive else path.glob(pattern)

        else:
            raise FileNotFoundError(f"Path '{path}' resolves to neither a file nor a directory.")

    def resolve(self, extension: str, *, recursive: bool = False) -> Iterable[Path]:
        """Resolve source to an iterable over files."""
        # normalize extension
        extension = extension.lower()
        if not extension.startswith("."):
            extension = "." + extension

        # so we can mutate
        source = self.source

        # if source is not a sequence
        if isinstance(source, (str, Path)):
            return self._expand_path(Path(source), extension, recursive)

        # if source is a sequence, get files from each path
        return chain.from_iterable(self._expand_path(Path(path), extension, recursive) for path in source)

    def to_struct(self) -> SourceStruct:
        """Returns a struct representation of the Source."""
        if isinstance(self.source, (str, Path)):
            return str(self.source)

        return [str(s) for s in self.source]

    @classmethod
    def from_struct(cls, struct: SourceStruct) -> Self:
        """Rebuilds Source from a struct."""
        if isinstance(struct, str):
            return cls(Path(struct))

        return cls([Path(s) for s in struct])
