# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import yaml
from pathlib import Path
from typing import Union, Any, Optional
from dotmap import DotMap
import tempfile

from .hash_utils import hash_directory


__all__ = ["IGConfig"]

class IGConfig:
    """
    Handles configuration loading, caching, and utility paths for the IceGraph pipeline.

    This class reads user and internal configuration files, provides structured access
    to relevant settings, computes input hashes for caching, and generates
    config files for external tools (e.g., `ml_suite`).
    """

    # constants
    PROGRAM_NAME = "icegraph"
    PROGRAM_VERSION = "0.1.2"

    def __init__(self, config_path: Union[str, Path]) -> None:
        """
        Initialize the Config object and set up necessary directory paths.

        Args:
            config_path (Union[str, Path]): Path to the user's main configuration file.
        """
        self.user_config_path = Path(config_path)

        # paths
        self.base_dir = Path(__file__).resolve().parent.parent.parent

        # config directories
        self.config_dir = self.base_dir / "config"
        self.internal_config_dir = self.config_dir / ".internal"  # internal-use configuration files
        self.user_config_dir = self.config_dir / "user"

        # internal config file paths
        self.feature_map_config_path = self.internal_config_dir / "features_map.yaml"

        # cache directory
        self.cache_dir = self.base_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # reference GCD files
        self.pass_2_gcd_path = Path(
            "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass2.i3.gz"
        )

        # cache attributes
        self._user_config_cache: DotMap | None = None
        self._feature_map_config_cache: dict | None = None
        self._input_hash_cache: str | None = None

    @property
    def user_config(self) -> DotMap:
        """
        Loads and returns the user configuration as a DotMap.

        Returns:
            DotMap: Parsed user configuration.
        """
        if self._user_config_cache is None:
            raw = self._load_file(self.user_config_path)
            self._user_config_cache = DotMap(raw)
        return self._user_config_cache

    @property
    def feature_map_config(self) -> DotMap:
        """
        Loads and returns the internal feature mapping configuration as a DotMap.

        Returns:
            DotMap: Parsed feature mapping configuration.
        """
        if self._feature_map_config_cache is None:
            raw = self._load_file(self.feature_map_config_path)
            self._feature_map_config_cache = DotMap(raw)
        return self._feature_map_config_cache

    @property
    def ml_suite_config_file(self) -> Path:
        """
        Writes a temporary YAML file containing ml_suite-compatible configuration
        and returns the path to that file.

        Returns:
            Path: Path to the temporary YAML config file.
        """
        feature_extraction_config: dict = self.user_config.feature_extraction.toDict()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            yaml.safe_dump(feature_extraction_config, tmp_file)
            return Path(tmp_file.name)

    def validate(self) -> bool:
        """
        Placeholder for configuration validation logic.

        Returns:
            bool: Whether the user configuration is valid.
        """
        ...

    @staticmethod
    def _load_file(path: Path) -> dict:
        """
        Load a YAML file into a dictionary.

        Args:
            path (Path): Path to the YAML file.

        Returns:
            dict: Parsed YAML content.
        """
        with path.open("r") as file:
            return yaml.safe_load(file)

    def get_input_state_hash(self) -> str:
            """
            Compute a consistent content-based hash of the input directory and configuration file.

            Returns:
                str: A hash representing the input state.
            """
            if self._input_hash_cache is None:
                input_dir = Path(self.user_config.input_dir)
                config_file = Path(self.user_config_path)
                self._input_hash_cache = hash_directory(input_dir, config_file, ".i3.zst")

            return self._input_hash_cache
