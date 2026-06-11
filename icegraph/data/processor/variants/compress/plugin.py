# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import pandas as pd

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

    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # grab from config
        by = env.resolve_cols(self.config.by)
        to = self.config.to
        out = self.config.out
        cols = env.resolve_cols(self.config.cols)
        dtype = self.config.dtype

        # ensure to is not the active frame
        if to == env.active:
            raise RuntimeError(
                f"`to` must differ from the active frame {env.active!r}: "
                f"compression reduces row count and cannot be merged back "
                f"onto its source."
            )

        # hstack phase: materialize each col as [R, N_i], concat to [R, sum N_i]
        parts = []
        widths = []
        for c in cols:
            series = main[c]

            # convert series to numpy
            if np.ndim(series.iloc[0]) == 0:
                arr = series.to_numpy(dtype=dtype).reshape(-1, 1)
            else:
                # raise if column is ragged
                try:
                    arr = np.asarray(series.tolist(), dtype=dtype)
                except ValueError as e:
                    raise ValueError(
                        f"Column {c!r} has inconsistent cell shapes: {e}"
                    ) from e
                if arr.ndim != 2:
                    raise ValueError(
                        f"Column {c!r} has inconsistent cell shapes (got {arr.ndim}D object "
                        f"array, expected 2D). All rows must share the same shape."
                    )

            # store array and width
            parts.append(arr)
            widths.append(arr.shape[1])

        # hstack
        values = np.hstack(parts)

        # build offset
        offset = np.concatenate(([0], np.cumsum(widths)))

        # vstack phase: group by key cols, gather rows of `values` per group
        grouped = main.groupby(by, sort=False, observed=True).indices
        keys = [k if isinstance(k, tuple) else (k,) for k in grouped.keys()]
        packed = [values[idx] for idx in grouped.values()]

        # build the output frame
        result = pd.DataFrame(keys, columns=by)
        result[out] = packed

        # record the compression
        if self.config.record_names:
            env.set_column_attr(out, "names", cols, domain=AttributeDomain.GLOBAL)
        if self.config.record_offset:
            env.set_column_attr(out, "offset", offset, domain=AttributeDomain.GLOBAL)

        # merge to frame and return
        return env.merge(result, to=to, on=by, validate="1:1")