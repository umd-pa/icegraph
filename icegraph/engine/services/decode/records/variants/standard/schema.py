# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from icegraph.common.record import Array, Scalar

__all__ = ["SimulationAttrs"]


class SimulationAttrs(BaseModel):
    """The simulation of one file, as the i3-simulation processor writes it."""

    sim_code:       Scalar[int]
    simulation:     Scalar[Literal["nugen", "corsika"]] = Field(alias="type")

    # generation column the sim code of each record is written to
    column:         Scalar[str]

    # the generation, as the columns of the one table it was read from
    weight_dict:    dict[str, Array] | None = None
    corsika_info:   dict[str, Array] | None = None
