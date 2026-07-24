# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from rich.pretty import pprint
import polars as pl

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.ui import console

from .config import InspectConfig

__all__ = ["Inspector"]


class Inspector(Processor[InspectConfig]):
    """Inspect the contents of the main dataframe."""
    name: ClassVar[str] = "inspect"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> InspectConfig:
        return InspectConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        main = item.main

        # print the dataframe info
        with pl.Config(tbl_cols=-1, tbl_width_chars=console.width):
            console.print(main.glimpse(return_type="string"))
            console.print(main.head())

        # print attrs
        attrs = item.attrs
        pprint(attrs, max_length=200)

        return item
