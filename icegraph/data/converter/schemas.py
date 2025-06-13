# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.config import IGConfig


__all__ = ["generate_vector_mapping"]

def generate_vector_mapping(config: IGConfig, invert: bool=False) -> dict[int, str] | dict[str, int]:
    requested_features = config.user_config.feature_extraction.feature_config.features
    feature_defs = config.feature_map_config.features.toDict()

    mapping = {}
    idx = 0

    for entry in requested_features:
        base_names = feature_defs[entry["class"]]
        kwargs = entry.get("kwargs")

        if not kwargs:
            for name in base_names:
                mapping[idx] = name
                idx += 1
        else:
            items = next(iter(kwargs.values()))
            for item in items:
                for name in base_names:
                    mapping[idx] = f"{name}_{item}"
                    idx += 1

    # values are unique, so inversion shouldn't be a problem
    if invert:
        return {v: k for k, v in mapping.items()}
    return mapping
