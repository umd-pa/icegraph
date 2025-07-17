# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Union, Any, Optional, Self
import tempfile

from dotmap import DotMap
import yaml
from pydantic import ValidationError

from .schemas import FullConfig

__all__ = ["IGConfig"]


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
        self.user_config_path = Path(config_path)

        # paths
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.src_dir = Path(__file__).resolve().parent.parent

        # config directories
        self.config_dir = self.base_dir / "config"
        self.internal_config_dir = self.src_dir / "config" / "defaults"  # internal-use configuration files

        # internal config file paths
        self.internal_config_file = self.internal_config_dir / "internal.config.yaml"

        # cache attributes
        self._user_config_cache: Optional[DotMap] = None
        self._internal_config_cache: Optional[DotMap] = None

        # fallback GCD file
        self.gcd_path = Path(
            self.user_config.io.gcd_path or
            "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass2.i3.gz"
        )

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
    def internal_config(self) -> DotMap:
        """
        Loads and returns the internal feature mapping configuration as a DotMap.

        Returns:
            DotMap: Parsed feature mapping configuration.
        """
        if self._internal_config_cache is None:
            raw = self._load_file(self.internal_config_file)
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
        feature_extraction_config: dict = self.user_config.feature_extraction.toDict()

        # ml_suite wants a config file, so we have to save a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            yaml.safe_dump(feature_extraction_config, tmp_file)
            return Path(tmp_file.name)

    def validate(self) -> None:
        """
        Input configuration validation.
        """
        try:
            raw = self._load_file(self.user_config_path)
            _ = FullConfig(**raw)
        except ValidationError as e:
            print(e)

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
    def register(cls, instance: Self) -> None:
        """Register the config instance for global access."""
        cls._instance = instance
        cls._instance.validate()

    @classmethod
    def get(cls) -> Self:
        """Get the globally accessible config instance."""
        if cls._instance is None:
            raise RuntimeError("Config not registered")
        return cls._instance
