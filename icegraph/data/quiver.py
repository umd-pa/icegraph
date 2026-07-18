# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator
from pathlib import Path
from collections.abc import Mapping
import shutil

import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd

__all__ = ["QuiverIPC"]


class QuiverIPC(Mapping[str, pa.Table]):

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __getitem__(self, key: str) -> pa.Table:
        f = self.root / f"{key}.arrow"
        if not f.exists():
            raise KeyError(key)
        return pa.ipc.open_file(pa.memory_map(str(f))).read_all()

    def __iter__(self) -> Iterator[str]:
        yield from sorted(
            p.relative_to(self.root).with_suffix("").as_posix()
            for p in self.root.rglob("*.arrow")
        )

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.arrow"))

    @classmethod
    def from_data(cls, data: dict[str, pd.DataFrame | pa.Table], root: str | Path) -> QuiverIPC:
        root = Path(root)

        # write each table
        for key, df in data.items():
            f = root / f"{key}.arrow"
            f.parent.mkdir(parents=True, exist_ok=True)
            feather.write_feather(df, f)

        return cls(root)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
