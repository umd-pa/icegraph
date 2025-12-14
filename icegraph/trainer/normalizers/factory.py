# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Dict, Type, Any, Union

import torch

from icegraph.trainer.base.exceptions import UnknownNormalizerError

# local subpackage imports
from .minmax import MinMaxNormalizer
from .zscore import ZScoreNormalizer
from .normalizer import Normalizer

__all__ = ["NormalizerFactory"]


class NormalizerFactory:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, model_cls: Type) -> None:
        """Register a model class under a given key."""
        cls._registry[model_cls.__name__] = model_cls

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> Normalizer:
        """
        Instantiate a registered model.
        Raises UnknownModelError if the name is unknown.
        """
        if name not in cls._registry:
            raise UnknownNormalizerError(f"Normalizer '{name}' is not registered; available: {list(cls._registry)}")
        return cls._registry[name](*args, **kwargs)


NormalizerFactory.register(MinMaxNormalizer)
NormalizerFactory.register(ZScoreNormalizer)
