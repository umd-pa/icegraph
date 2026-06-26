# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path
import subprocess
import atexit
from contextlib import suppress

from torch.utils.tensorboard import SummaryWriter

from icegraph.utils import is_port_available

__all__ = ["TensorBoardService"]

# module logger
import logging
logger = logging.getLogger(__name__)


class TensorBoardService:
    """
    A utility class for managing a TensorBoard instance and writing logs.
    """

    def __init__(self, log_dir: str | Path, *, port: int = 6006):
        """
        Initialize the TensorBoardService logger.

        Args:
            log_dir (Union[str, Path]): The directory where TensorBoardService logs will be stored.
        """
        self.log_dir = log_dir

        # init tensorboard
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

        # cache port
        self._port = port

        if not is_port_available(self._port):
            raise RuntimeError(
                f"Failed to launch TensorBoardService. "
                f"Specified port {self._port} is already being used by another process."
            )

        self._process: subprocess.Popen | None = None

        # register atexit shutdown
        atexit.register(self.close)

    def __del__(self) -> None:
        """
        Destructor that ensures TensorBoardService is shut down on object deletion.
        """
        with suppress(Exception):
            self.close()

    @property
    def writer(self) -> SummaryWriter:
        """
        Get the internal SummaryWriter instance.

        Returns:
            SummaryWriter: The PyTorch TensorBoard SummaryWriter.
        """
        return self._writer

    def launch(self) -> tuple[int, int]:
        """
        Launch a TensorBoard instance.
        """
        self._process = subprocess.Popen([
            "tensorboard",
            "--logdir", str(self.log_dir),
            "--port", str(self._port)
        ])

        logger.info(
            f"%s started with PID=%d at http://localhost:%d",
            type(self).__name__, self._process.pid, self._port
        )

        return self._process.pid, self._port

    def close(self) -> None:
        """Shut down any active TensorBoardService instance if running."""
        with suppress(Exception):
            self._writer.close()

        proc = self._process
        self._process = None  # clear early to avoid double-shutdown races
        if proc is None:
            return

        pid = proc.pid

        # if already exited, just clear
        if proc.poll() is not None:
            self._process = None
            logger.debug(
                "%s already exited (PID=%d)", type(self).__name__, pid
            )
            return

        try:
            proc.terminate()
            proc.wait(timeout=3)
            logger.info(
                "%s terminated (PID=%d, port=%d)", type(self).__name__, pid, self._port
            )
        except subprocess.TimeoutExpired:
            with suppress(Exception):
                proc.kill()
            with suppress(Exception):
                proc.wait(timeout=3)
            logger.warning(
                "%s killed after timeout (PID=%d, port=%d)", type(self).__name__, pid, self._port
            )
        except Exception:
            logger.exception(
                "failed to shutdown %s (PID=%d, port=%d)",type(self).__name__, pid, self._port
            )