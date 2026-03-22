# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from rich.pretty import pprint
import pandas as pd

from icegraph.data.processor import Processor
from icegraph.data.types import Envelope
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

    def _process(self, env: Envelope) -> Envelope | None:
        main = env.main

        # print the dataframe info
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)  # auto-detect terminal width (no wrap)
        pd.set_option("display.expand_frame_repr", False)  # prevent wrapping to multiple lines

        main.info(show_counts=True)
        console.print(main.head())

        # print attrs
        attrs = env.attrs
        pprint(attrs, max_length=200)

        return env
