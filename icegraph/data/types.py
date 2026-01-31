# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Any
import weakref
import shutil
import time

import portalocker
import pandas as pd



@dataclass
class FileHandle:
    """
    A closeable handle to a file. On `close()` (or context exit), the file is deleted.
    Also tries to remove the parent directory if it becomes empty.

    Works within a single owner/stage at a time.
    """
    src: Path
    _finalizer: weakref.finalize = field(init=False, repr=False)

    FILELOCK_SH_TIMEOUT: ClassVar[int] = 30
    FILELOCK_EX_TIMEOUT: ClassVar[int] = 60

    def __post_init__(self) -> None:
        """Install a finalizer that cleans up the file path on GC."""
        self._finalizer = weakref.finalize(self, FileHandle._cleanup, self.src)

    @property
    def _lock_path(self) -> Path:
        return self.src.with_suffix(self.src.suffix + ".lock")

    def lock_shared(self, timeout: int = FILELOCK_SH_TIMEOUT):
        """Readers: allow many; blocks if a writer holds the lock."""
        return portalocker.Lock(
            self._lock_path, mode="a+", timeout=timeout,
            flags=portalocker.LockFlags.SHARED | portalocker.LockFlags.NON_BLOCKING
        )

    def lock_exclusive(self, timeout: int = FILELOCK_EX_TIMEOUT):
        """Writers: exclusive; blocks until all readers/writers are done."""
        return portalocker.Lock(
            self._lock_path, mode="a+", timeout=timeout,
            flags=portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
        )

    def close(self) -> None:
        """
        Delete the underlying file or directory (best-effort).

        This does not remove the sidecar ``.lock``; call :meth:`remove_lock`
        after the lock context exits to unlink it safely on all platforms.
        """
        if self._finalizer.alive:
            self._finalizer()

    @staticmethod
    def _cleanup(path: Path) -> None:
        """Finalizer target that removes the file path or directory tree."""
        try:
            path.unlink(missing_ok=True)
        except IsADirectoryError:
            shutil.rmtree(path, ignore_errors=True)

    def remove_lock(self, retries: int = 5, delay: float = 0.05) -> None:
        """
        Remove the sidecar ``.lock`` after the lock context has exited.

        Retries are helpful on Windows where the handle can linger briefly.

        Args:
            retries: Maximum retry attempts.
            delay: Seconds to sleep between retries.
        """

        lock_path = self._lock_path
        for _ in range(max(1, retries)):
            try:
                lock_path.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(delay)
            except FileNotFoundError:
                # no file, break
                return
            except OSError:
                # Best effort; retry a few times
                time.sleep(delay)

@dataclass
class Envelope:
    """
    Data + metadata wrapper exchanged between stages.

    Attributes:
        df: The payload DataFrame.
        fh: The temporary file handle associated with this item.
        metrics: Dict storing per stage metrics.
        attrs: Arbitrary nested attributes (auto-nesting).
    """
    df: pd.DataFrame
    fh: FileHandle
    metrics: dict[int, float] = field(default_factory=dict)
    global_attrs: dict[str, Any] = field(default_factory=nested_dict)
    local_attrs: dict[str, Any] = field(default_factory=nested_dict)

    def finalize(self) -> Self:
        # Close the temp file under locks, then yield self
        with self.fh.lock_exclusive():
            self.fh.close()
        # remove the lock file
        self.fh.remove_lock()

        return self