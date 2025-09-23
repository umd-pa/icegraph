# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from pathlib import Path
import uuid
import shutil
from contextlib import contextmanager

import portalocker

from icegraph.config import IGConfig

__all__ = ["MPTempDir"]


class MPTempDir:
    """
    Provides a shared cache directory at ~/.cache/<PROGRAM_NAME> and
    per-file concurrency control.
    """

    def __init__(self) -> None:
        self._config = IGConfig.get()
        self._lock_timeout = 30

        # make the temp_dir
        self.tempdir = self._get_tempdir()
        self.tempdir.mkdir(parents=True, exist_ok=True)

        # directory lock marker (keeps dir alive while any instance is in use)
        self._dir_lock = self.tempdir / f"in-use.{uuid.uuid4().hex}.lock"
        self._dir_lock.touch(exist_ok=True)

    def terminate_instance(self) -> None:
        # remove the instance marker
        try:
            os.remove(self._dir_lock)
        except FileNotFoundError:
            pass

        # if no other instance markers remain, clean up directory
        try:
            has_other_users = any(
                p.name.startswith("in-use") for p in self.tempdir.glob("*.lock")
            )
            if not has_other_users:
                shutil.rmtree(self.tempdir)
        except FileNotFoundError:
            pass

    def _get_tempdir(self) -> Path:
        return Path.home() / ".cache" / self._config.PROGRAM_NAME

    @staticmethod
    def _lockfile_for(data_path: Path) -> Path:
        return data_path.with_name(data_path.name + ".lock")

    @contextmanager
    def shared_read_lock(self, data_path: Path):
        lock_path = self._lockfile_for(data_path)
        with portalocker.Lock(
            lock_path,
            mode="a+",
            flags=portalocker.LockFlags.SHARED | portalocker.LockFlags.NON_BLOCKING,
            timeout=self._lock_timeout
        ):
            yield

    @contextmanager
    def exclusive_write_lock(self, data_path: Path):
        lock_path = self._lockfile_for(data_path)
        with portalocker.Lock(
            lock_path,
            mode="a+",
            flags=portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
            timeout=self._lock_timeout
        ):
            yield
