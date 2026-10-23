# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import StrEnum

__all__ = ["ComponentKind"]


class ComponentKind(StrEnum):
    NORMALIZER  = "normalizer"
    MODEL       = "model"
    TRANSFORMER = "transformer"
    EDGES       = "edges"
    OPTIMIZER   = "optimizer"
    LOSS        = "loss"

    @classmethod
    def all(cls) -> tuple[ComponentKind, ...]:
        return tuple(cls)

    @classmethod
    def inference(cls) -> tuple[ComponentKind, ...]:
        """Kinds an inference run reconstructs from an exported model.

        The exporter writes the config and state of exactly these, so both sides
        of the round trip stay in step.
        """
        return cls.MODEL, cls.NORMALIZER, cls.TRANSFORMER, cls.EDGES
