# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import json
import os
import time
from typing import Optional
import xxhash

from icegraph import config


class I3DirectoryCacheHandler:
    # cache file should be invalidated on version update to ensure proper function
    _cache_file = os.path.join(config.CACHE_DIR, f".conversion_cache.{config.PROGRAM_VERSION}.json")
    _expiration_time = 7 * 24 * 60 * 60  # 7 days

    @classmethod
    def _load_cache(cls) -> dict:
        if not os.path.exists(cls._cache_file):
            return {}
        with open(cls._cache_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    @classmethod
    def _save_cache(cls, cache: dict) -> None:
        with open(cls._cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    @classmethod
    def _hash_directory(cls, input_dir: str, config_path: str) -> str:
        """Generate a hash based on top-level .i3 file names/contents and a config file."""
        h = xxhash.xxh64()

        # Hash all top-level .i3 files in the directory
        for file in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, file)
            if os.path.isfile(path) and file.endswith(".i3.zst"):
                h.update(file.encode())  # include file name
                with open(path, "rb") as f:
                    while chunk := f.read(8192):
                        h.update(chunk)

        # Hash config file name and content
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        h.update(os.path.basename(config_path).encode())
        with open(config_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)

        return h.hexdigest()

    @classmethod
    def register(cls, input_dir: str, output_dir: str, config_path: str) -> None:
        dir_hash = cls._hash_directory(input_dir, config_path)
        cache = cls._load_cache()
        cache[dir_hash] = {
            "converted_path": output_dir,
            "timestamp": time.time()
        }
        cls._save_cache(cache)

    @classmethod
    def query(cls, input_dir: str, config_path: str) -> Optional[str]:
        dir_hash = cls._hash_directory(input_dir, config_path)
        cache = cls._load_cache()
        entry = cache.get(dir_hash)

        if not entry:
            return None

        converted_path = entry["converted_path"]
        timestamp = entry["timestamp"]

        if not os.path.exists(converted_path) or (time.time() - timestamp > cls._expiration_time):
            del cache[dir_hash]
            cls._save_cache(cache)
            return None

        return converted_path

    @classmethod
    def get_hash(cls, input_dir: str, config_path: str) -> str:
        return cls._hash_directory(input_dir, config_path)

    @classmethod
    def clear_expired(cls) -> None:
        cache = cls._load_cache()
        now = time.time()
        new_cache = {
            k: v for k, v in cache.items()
            if os.path.exists(v["converted_path"]) and (now - v["timestamp"] <= cls._expiration_time)
        }
        cls._save_cache(new_cache)

    @classmethod
    def clear_all(cls) -> None:
        if os.path.exists(cls._cache_file):
            os.remove(cls._cache_file)

