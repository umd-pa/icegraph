 # Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import polars as pl

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.common.data import AttributeDomain

from .config import CompressorConfig

__all__ = ["Compressor"]


class Compressor(Processor[CompressorConfig]):
    """Concatenate columns and stack rows into per-group 2D arrays.

    Equivalent to a per-row hstack of cols followed by a per-group vstack.

    For each group, the output cell has shape ``[R,sum(N_i)]`` where R is the
    number of rows in the group and N_i is the feature width of input column i
    (1 for scalar columns).
    """
    name: ClassVar[str] = "compress"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CompressorConfig:
        return CompressorConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # grab from config
        by = item.resolve_cols(self.config.by)
        to = self.config.to
        out = str(self.config.out)
        cols = item.resolve_cols(self.config.cols)
        dtype = self.config.dtype

        # ensure to is not the active frame
        if to == item.active:
            raise RuntimeError(
                f"`to` must differ from the active frame {item.active!r}: "
                f"compression reduces row count and cannot be merged back "
                f"onto its source."
            )

        # hstack phase: materialize each col as [R, N_i], concat to [R, sum N_i]
        parts = []
        widths = []
        for c in cols:
            series = main.get_column(c)

            # convert series to numpy
            if isinstance(series.dtype, (pl.List, pl.Array)):
                # raise if column is ragged
                try:
                    arr = np.asarray(series.to_list(), dtype=dtype)
                except ValueError as e:
                    raise ValueError(
                        f"Column {c!r} has inconsistent cell shapes: {e}"
                    ) from e
                if arr.ndim != 2:
                    raise ValueError(
                        f"Column {c!r} has inconsistent cell shapes (got {arr.ndim}D object "
                        f"array, expected 2D). All rows must share the same shape."
                    )
            else:
                arr = series.to_numpy()
                if dtype is not None:
                    arr = arr.astype(dtype, copy=False)
                arr = arr.reshape(-1, 1)

            # store array and width
            parts.append(arr)
            widths.append(arr.shape[1])

        # hstack
        values = np.hstack(parts)

        # build offset
        offset = np.concatenate(([0], np.cumsum(widths)))

        # vstack phase: group by key cols, gather rows of `values` per group
        grouped = (
            main
            .with_row_index("__row_idx__")
            .group_by(by, maintain_order=True)
            .agg(pl.col("__row_idx__"))
        )
        packed = [
            values[np.asarray(idx, dtype=np.int64)]
            for idx in grouped.get_column("__row_idx__").to_list()
        ]

        # build the output frame
        inner = pl.Series(values=values.reshape(-1)[:0]).dtype  # numpy dtype -> polars dtype
        col_dtype = pl.List(pl.Array(inner, values.shape[1]))
        result = grouped.drop("__row_idx__").with_columns(
            pl.Series(out, packed, dtype=col_dtype)
        )

        # record the compression
        if self.config.record_names:
            item.set_column_attr(out, "names", cols, domain=AttributeDomain.GLOBAL)
        if self.config.record_offset:
            item.set_column_attr(out, "offset", offset, domain=AttributeDomain.GLOBAL)

        # merge to frame and return
        return item.merge(result, to=to, on=by, validate="1:1")
