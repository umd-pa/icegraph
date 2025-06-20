# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import json
import time
from typing import Optional, Union, Type, TypeVar, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import numpy as np
import shutil
from filelock import FileLock, BaseFileLock

from icegraph.config import IGConfig
from icegraph.console import Console
from icegraph.data.cache.exceptions import CacheBuildFailure, InvalidCache, CacheRegistrationError

__all__ = ["IGConverterCache", "IGDataCache"]


tCacheEntry = Dict[str, Any]

if TYPE_CHECKING:
    from icegraph.data.base import IGData

t = TypeVar("t", bound="IGData")

class IGConverterCache:
    """
    Cache handler for storing and retrieving I3 dataset conversion outputs.

    Stores a mapping from input-state-hash to converted output directory with a timestamp.
    Entries expire after a fixed TTL.
    """
    def __init__(self, config: IGConfig) -> None:
        """
        Initialize the converter-level cache.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
        """
        self._config: IGConfig = config
        self._cache_file: Path = (
            self._config.cache_dir / "converter" /
            f".converter_cache_{self._config.PROGRAM_VERSION}.json"
        )
        self._expiration_time: float = 7 * 24 * 60 * 60  # 7 days

        # ensure directory exists
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CacheBuildFailure(f"Failed to create converter cache directory: {e}")

    def _load_cache(self) -> Dict[str, tCacheEntry]:
        """
        Load the cache from disk. Returns an empty dict if missing or invalid.
        """
        if not self._cache_file.exists():
            return {}
        try:
            text = self._cache_file.read_text(encoding="utf-8")
            return json.loads(text)
        except (json.JSONDecodeError, OSError) as e:
            raise CacheBuildFailure(f"Error loading converter cache: {e}")

    def _save_cache(self, cache: Dict[str, tCacheEntry]) -> None:
        """
        Save the cache mapping to disk as JSON.
        """
        try:
            self._cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError as e:
            raise CacheBuildFailure(f"Error saving converter cache: {e}")

    def register(self, output_dir: Union[str, Path]) -> None:
        """
        Register a new conversion output in the cache.

        Args:
            output_dir (Union[str, Path]): Path to the output directory.
        """
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
        Query for a previously converted output.

        Returns:
            Optional[Path]: Path to converted output if valid and not expired; otherwise None.
        """
        dir_hash = self._config.get_input_state_hash()
        cache = self._load_cache()
        entry = cache.get(dir_hash)

        if not entry:
            return None

        try:
            converted_path = Path(entry["converted_path"])
            timestamp = float(entry.get("timestamp", 0))

        except (TypeError, ValueError) as e:
            raise InvalidCache(f"Invalid cache entry format: {e}")

        if not converted_path.exists() or (time.time() - timestamp > self._expiration_time):
            cache.pop(dir_hash, None)
            self._save_cache(cache)
            return None

        return converted_path

    def clear_expired(self) -> None:
        """
        Remove expired entries based on TTL and path existence.
        """

        cache = self._load_cache()
        now = time.time()
        updated = False

        for k, entry in list(cache.items()):
            path = Path(entry.get("converted_path", ""))
            ts = entry.get("timestamp", 0)

            if not path.exists() or (now - ts > self._expiration_time):
                cache.pop(k, None)
                updated = True

        if updated:
            self._save_cache(cache)

    def clear_all(self) -> None:
        """
        Delete the entire converter cache file.
        """
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
        except OSError as e:
            Console.out(f"Error clearing converter cache: {e}", severity=2)


class IGDataCache:
    """
    Per-event cache handler for storing feature/label arrays on disk.

    Each event is stored as an .npz file under a subset-named subdirectory.
    """
    def __init__(self, config: IGConfig) -> None:
        """
        Initialize data cache.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
        """
        self._config: IGConfig = config
        self._expiration_time: float = 7 * 24 * 60 * 60  # 7 days

        input_state_hash = config.get_input_state_hash()
        self.cache_dir: Path = config.cache_dir / "data" / input_state_hash
        self.metadata_file: Path = self.cache_dir / ".metadata.json"

        self.lock: BaseFileLock = FileLock(str(self.cache_dir / f"{input_state_hash}.lock"), timeout=10)

        # Ensure on-disk structure and metadata
        self._setup_cache()
        self.build_required: bool = True

        # Helper to form target file paths
        self._target_file = lambda subset, idx: self.cache_dir / subset / f"{idx}.npz"

    def _setup_cache(self) -> None:
        """
        Create cache directory and write metadata atomically.
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            metadata = {"version": self._config.PROGRAM_VERSION}

            tmp = self.metadata_file.with_suffix(self.metadata_file.suffix + ".tmp")
            tmp.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
            tmp.replace(self.metadata_file)

        except OSError as e:
            raise CacheBuildFailure(f"Failed to initialize data cache: {e}")

    def reset(self) -> None:
        """
        Clear all cached event files and reinitialize metadata.
        """
        try:
            shutil.rmtree(self.cache_dir)

        except OSError:
            pass

        self._setup_cache()
        self.build_required = True

    def _validate(self, expected_size: int) -> bool:
        """
        Confirm that the number of .npz files matches expectation.
        """
        file_count = sum(1 for _ in self.cache_dir.glob("*/*.npz"))
        return expected_size == file_count

    def check(self, expected_size: int) -> bool:
        """
        Validate cache metadata and content size.

        Returns:
            bool: True if cache is valid; False otherwise.
        """
        # Metadata existence
        try:
            stat = self.metadata_file.stat()

        except FileNotFoundError:
            return False

        # Metadata parsing
        try:
            data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
            if data.get("version") != self._config.PROGRAM_VERSION:
                return False

            if (time.time() - stat.st_mtime) > self._expiration_time:
                return False

        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            Console.out(f"Cache metadata invalid or corrupted: {e}", severity=2)
            return False

        # Content validation
        if not self._validate(expected_size):
            return False

        self.build_required = False
        return True

    def register(self, data_subclass: Type[t], idx: int, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Save features and labels for one event to disk.

        Args:
            data_subclass (Type[IGData]): The dataset class (provides subset name).
            idx (int): Event index.
            features (np.ndarray): Feature array.
            labels (np.ndarray): Label array.
        """
        target = self._target_file(data_subclass.subset, idx)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, f=features, l=labels)

        except OSError as e:
            raise CacheRegistrationError(f"Failed to write cache file {target}: {e}")

    def query(self, data_subclass: Type[t], idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels for one event from cache.

        Args:
            data_subclass (Type[IGData]): The dataset class (provides subset name).
            idx (int): Event index.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (features, labels)

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        target = self._target_file(data_subclass.subset, idx)
        try:
            data = np.load(target)
            return data["f"], data["l"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cache missing for subset={data_subclass.subset}, idx={idx}. "
                "Cache may need to be rebuilt."
            )
