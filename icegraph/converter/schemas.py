# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import yaml


class MLSuiteVectorMapping:

    def __init__(self, extraction_config_path: str, features_map_config_path: str) -> None:
        self.extraction_config_path = extraction_config_path
        self.features_map_config_path = features_map_config_path

        self.config = {}
        self._load_configs()

    def _load_configs(self) -> None:
        # load the ml_suite feature extraction configuration file
        with open(self.extraction_config_path, "r") as file:
            self.config["extraction"] = yaml.safe_load(file)

        # load the ml_suite features map
        with open(self.features_map_config_path) as file:
            self.config["features_map"] = yaml.safe_load(file)

    def get_mapping(self) -> dict[int, str]:
        requested_features = self.config["extraction"]["feature_config"]["features"]
        feature_defs = self.config["features_map"]["features"]

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
