# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# base normalizer
from .normalizer import Normalizer

# implementations
from .zscore import ZScoreNormalizer
from .minmax import MinMaxNormalizer

__all__ = ["ZScoreNormalizer", "MinMaxNormalizer", "resolve_normalizer", "Normalizer"]

_REGISTRY: dict[str, type[Normalizer]] = {
    "ZScoreNormalizer": ZScoreNormalizer,
    "MinMaxNormalizer": MinMaxNormalizer,
}

def resolve_normalizer(name: str, **kwargs) -> Normalizer:
    try:
        cls = _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown normalizer '{name}'. Available: {', '.join(_REGISTRY)}") from e
    return cls(**kwargs)
