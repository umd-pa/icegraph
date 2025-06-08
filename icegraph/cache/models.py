# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import json
import time
from typing import Optional, Union
from pathlib import Path

from icegraph.config import Config


class I3ConversionCache:
    def __init__(self, config: Config):
        """
        Initialize a cache handler for storing and retrieving I3 dataset conversion outputs.

        Args:
            config (Config): A config object providing user paths and constants.
        """
        self._config: Config = config
        self._cache_file = self._config.cache_dir / f".conversion_cache.{self._config.PROGRAM_VERSION}.json"
        self._expiration_time = 7 * 24 * 60 * 60  # 7 days

    def _load_cache(self) -> dict:
        """
        Load the cache from disk. Returns an empty dict if the file doesn't exist or is invalid.

        Returns:
            dict: Cached hash-to-path mappings.
        """
        if not self._cache_file.exists():
            return {}
        try:
            return json.loads(self._cache_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_cache(self, cache: dict) -> None:
        """
        Save the provided cache dictionary to disk as JSON.

        Args:
            cache (dict): The cache mapping to save.
        """
        self._cache_file.write_text(json.dumps(cache, indent=2))

    def register(self, output_dir: Union[str, Path]) -> None:
        """
        Register a new conversion output in the cache.

        Args:
            output_dir (Union[str, Path]): Path to the output directory.
        """

        # normalize
        output_dir = Path(output_dir)

        dir_hash = self._config.get_input_state_hash()
        cache = self._load_cache()
        cache[dir_hash] = {
            "converted_path": str(output_dir),
            "timestamp": time.time()
        }
        self._save_cache(cache)

    def query(self) -> Optional[Path]:
        """
        Query the cache for a matching converted output.

        Returns:
            Optional[Path]: Path to converted output, or None if not cached or expired.
        """

        dir_hash = self._config.get_input_state_hash()
        cache = self._load_cache()
        entry = cache.get(dir_hash)

        if not entry:
            return None

        converted_path = Path(entry["converted_path"])
        timestamp = entry["timestamp"]

        if (
                not isinstance(timestamp, (int, float))
                or not converted_path.exists()
                or (time.time() - timestamp > self._expiration_time)
        ):
            del cache[dir_hash]
            self._save_cache(cache)
            return None

        return converted_path

    def clear_expired(self) -> None:
        """
        Remove any expired entries from the cache based on file existence and timestamp.
        """
        cache = self._load_cache()
        now = time.time()
        new_cache = {
            k: v for k, v in cache.items()
            if Path(v["converted_path"]).exists() and (now - v["timestamp"] <= self._expiration_time)
        }
        self._save_cache(new_cache)

    def clear_all(self) -> None:
        """
        Delete the entire cache file from disk.
        """
        if self._cache_file.exists():
            self._cache_file.unlink()
