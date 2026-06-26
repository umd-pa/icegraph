# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icegraph.common.factory import Factory

from .module import TransformerModule

from . import modules

__all__ = ["TransformerModuleFactory"]


class TransformerModuleFactory(Factory[TransformerModule]):
    pass


for name in modules.__all__:
    TransformerModuleFactory.register(getattr(modules, name))
