# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.config import Config


__all__ = ["MLSuiteVectorMapping"]

class MLSuiteVectorMapping:
    """
    Generates a mapping from vector indices to feature names for ml_suite feature extraction.

    Uses the user's feature extraction configuration and the internal feature map to construct
    a flat index-to-name mapping.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the mapping generator with configuration data.

        Args:
            config (Config): IceGraph configuration object containing user settings.
        """
        self._config: Config = config

    def get_mapping(self) -> dict[int, str]:
        """
        Generate a mapping from feature vector indices to human-readable feature names.

        The returned dictionary maps each index (as used in the vectorized feature table)
        to its corresponding feature name, including item suffixes when needed.

        Returns:
            dict[int, str]: A mapping from vector indices to column names.
        """
        requested_features: list = self._config.user_config.feature_extraction.feature_config.features
        feature_defs: dict = self._config.feature_map_config.features.toDict()

        mapping: dict[int, str] = {}
        idx = 0

        for entry in requested_features:
            class_name = entry["class"]
            base_names = feature_defs[class_name]

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

        return mapping

