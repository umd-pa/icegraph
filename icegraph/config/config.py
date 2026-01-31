# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, Self, Dict
import tempfile
import os

from dotmap import DotMap
import yaml
from pydantic import ValidationError

from ._schemas import FullConfig

__all__ = ["IGConfig"]

# module logger
import logging
logger = logging.getLogger(__name__)


class IGConfig:
    """
    Handles configuration loading, caching, and utility paths for the IceGraph pipeline.

    This class reads user and internal configuration files, provides structured access
    to relevant settings, and generates config files for external tools (e.g., `ml_suite`).
    """

    _instance: Optional[Self] = None

    PROGRAM_NAME = "icegraph"

    def __init__(self, config_path: Union[str, Path]) -> None:
        """
        Initialize the Config object and set up necessary directory paths.

        Args:
            config_path (Union[str, Path]): Path to the user's main configuration file.
        """
        # cache attributes
        self._user_config_cache: Optional[DotMap] = None
        self._internal_config_cache: Optional[DotMap] = None

        self.paths = {"user_config": Path(config_path)}
        self.paths.update(self._load_paths())

    def _load_paths(self) -> Dict[str, Path]:

        src_path = Path(__file__).resolve().parent.parent

        paths: Dict[str, Path] = {
            "base": Path(__file__).resolve().parent.parent.parent,
            "src": src_path,
            "internal_config": src_path / "config" / "defaults"/ "internal.config.yaml",
            "gcd": Path(
                self.user_config.io.gcd_path or
                "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass2.i3.gz"
            )
        }

        return paths

    @staticmethod
    def get_xdg_cache_dir() -> Path:
        return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "icegraph"

    @property
    def user_config(self) -> DotMap:
        """
        Loads and returns the user configuration as a DotMap.

        Returns:
            DotMap: Parsed user configuration.
        """
        if self._user_config_cache is None:
            raw = self._load_file(self.paths["user_config"])
            self._user_config_cache = DotMap(raw)
        return self._user_config_cache

    @property
    def internal_config(self) -> DotMap:
        """
        Loads and returns the internal feature mapping configuration as a DotMap.

        Returns:
            DotMap: Parsed feature mapping configuration.
        """
        if self._internal_config_cache is None:
            raw = self._load_file(self.paths["internal_config"])
            self._internal_config_cache = DotMap(raw)
        return self._internal_config_cache

    @property
    def ml_suite_config_file(self) -> Path:
        """
        Writes a temporary YAML file containing ml_suite-compatible configuration
        and returns the path to that file.

        Returns:
            Path: Path to the temporary YAML config file.
        """
        feature_extraction_config: dict = self.ml_suite_config

        # ml_suite wants a config file, so we have to save a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            yaml.safe_dump(feature_extraction_config, tmp_file)
            return Path(tmp_file.name)

    @property
    def ml_suite_config(self) -> Dict:
        """Load the configuration for ML suite."""
        return self.user_config.feature_extraction.toDict()

    def validate(self) -> None:
        """
        Input configuration validation.
        """
        try:
            raw = self._load_file(self.paths["user_config"])
            FullConfig(**raw)
        except ValidationError:
            logging.exception("config validation failed", exc_info=True)
            raise

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

    @classmethod
    def register(cls, instance: IGConfig) -> None:
        """Register the config instance for global access."""
        cls._instance = instance
        cls._instance.validate()

        logger.debug(
            "registered user config for %s: %s",
            type(instance).__name__,
            instance.user_config.toDict()
        )

    @classmethod
    def get(cls) -> Self:
        """Get the globally accessible config instance."""
        if cls._instance is None:
            raise RuntimeError("Config not registered")
        return cls._instance
