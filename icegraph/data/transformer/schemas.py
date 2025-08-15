# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.config import IGConfig
from .base.exceptions import VectorMappingError

__all__ = ["generate_vector_mapping"]


def generate_vector_mapping(invert: bool=False) -> dict[int, str] | dict[str, int]:
    """
    Build a mapping between flat vector indices and feature names for ml_suite generated tables.

    - If `invert=False`, returns a dict mapping each integer index to feature-name (str).
    - If `invert=True`, returns the reverse mapping, feature-name to index.

    Raises:
        VectorMappingError
    """
    # load config instance
    config = IGConfig.get()
    requested = config.user_config.feature_extraction.feature_config.features
    feature_defs = config.internal_config.features.toDict()

    if not requested:
        raise VectorMappingError("No features requested")

    mapping: dict[int, str] = {}
    idx = 0

    for entry in requested:
        cls = entry.get("class")
        if cls not in feature_defs:
            raise VectorMappingError(f"Unknown feature class '{cls}'")
        base_names = feature_defs[cls]
        if not base_names:
            raise VectorMappingError(f"No base names for class '{cls}'")

        kwargs = entry.get("kwargs")
        if kwargs and (not isinstance(kwargs, dict) or
                       any(not hasattr(v, "__iter__") for v in kwargs.values())):
            raise VectorMappingError(f"Invalid kwargs for '{cls}': {kwargs!r}")

        if not kwargs:
            for name in base_names:
                if idx in mapping:
                    raise VectorMappingError(f"Index collision at {idx}")
                mapping[idx] = name
                idx += 1
        else:
            items = next(iter(kwargs.values()))
            if not items:
                raise VectorMappingError(f"No items for parametrized class '{cls}'")
            for item in items:
                for name in base_names:
                    if idx in mapping:
                        raise VectorMappingError(f"Index collision at {idx}")
                    mapping[idx] = f"{name}_{item}"
                    idx += 1

    if invert:
        inv = {v: k for k, v in mapping.items()}
        if len(inv) != len(mapping):
            raise VectorMappingError("Mapping values are not unique; cannot invert")
        return inv

    return mapping
