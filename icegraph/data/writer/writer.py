# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, TypeVar, Any
from datetime import datetime
import math
import shutil

import polars as pl

from icegraph import __version__
from icegraph.utils.hashutils import CBORBlake2B
from icegraph.common.data import AttributeDomain, flatten

from ..stage import Stage
from ..envelope import Envelope

import logging
logger = logging.getLogger(__name__)

__all__ = ["Writer"]


C = TypeVar("C")


class Writer(Stage[C, Envelope]):
    """Base class for pipeline DataFrame writers."""
    suffix: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # ensure suffix is defined for all subclasses
        if getattr(cls, "suffix", None) is None:
            raise RuntimeError(f"Writer {cls.__name__} must implement the class variable 'suffix'")

    def on_attach(self) -> None:
        # ensure a directory was provided for the writer
        if self._ctx.outdir is None:
            raise ValueError(f"{type(self).__name__} requires an output directory provided via the attach context.")

    # provide a convenient property to access the outdir
    @cached_property
    def outdir(self) -> Path:
        assert self._ctx.outdir is not None
        return self._ctx.outdir

    def _process(self, item: Envelope) -> Envelope | None:
        # ensure no doubly ragged columns at this point
        for col, dt in item.main.schema.items():
            if isinstance(dt, pl.List) and isinstance(dt.inner, pl.List):
                raise ValueError(
                    f"{col}: {dt} - nested List is unsupported. Use List(Array(...)) for fixed-width inners, "
                    "or flatten the outer level into a separate column."
                )

        # compute id and set id
        hasher = CBORBlake2B()

        # take hash of the data itself for file ID
        _id = hasher(item.main.rows())

        # take hash of global attributes for set ID
        _set_id = hasher(item.attrs[AttributeDomain.GLOBAL.name])

        # register ids as attrs
        item.set_local_attr("id", _id)
        item.set_global_attr("set_id", _set_id)

        # construct metadata
        metadata: dict[str, Any] = {
            "info": {
                "timestamp": datetime.now().timestamp(),
                "icegraph": {
                    "version": __version__
                },
                "writer": {
                    "name": type(self).name,
                    "version": type(self).version
                }
            }
        }

        # include env metadata
        metadata.update(item.attrs)

        # add entry count to metadata
        metadata["entries"] = item.main.height

        # make metadata serializable
        metadata = flatten(metadata)

        # generate output file path
        origin = Path(item.get_local_attr("origin"))
        fp = self.outdir / origin.with_suffix(type(self).suffix).name

        # ensure no stale keys
        if fp.exists():
            try:
                if fp.is_dir():
                    shutil.rmtree(fp)
                    logger.debug(f"removed directory at {fp!s}")
                elif fp.exists():
                    fp.unlink()
                    logger.debug(f"unlinked file at {fp!s}")
            except OSError as e:
                raise RuntimeError(f"Failed to remove existing file: {fp}") from e

        # delegate write to subclass
        self._write(item.main, metadata, fp)

        return item

    @abstractmethod
    def _write(self, table: pl.DataFrame, metadata: dict[str, Any], fp: Path) -> None:
        ...