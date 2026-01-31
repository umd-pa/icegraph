# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional, Tuple
from pathlib import Path
import subprocess
import atexit
from contextlib import suppress
import signal

from torch.utils.tensorboard import SummaryWriter

from icegraph.config import IGConfig
from icegraph.console._streams import suppress_stderr
from icegraph.utils import is_port_available

__all__ = ["TensorBoardService"]

# module logger
import logging
logger = logging.getLogger(__name__)


class TensorBoardService:
    """
    A utility class for managing a TensorBoard instance and writing logs.
    """

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize the TensorBoardService logger.

        Args:
            log_dir (Union[str, Path]): The directory where TensorBoardService logs will be stored.
        """
        self.log_dir = log_dir

        # init tensorboard
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

        # grab tensorboard config
        self.port = IGConfig.get().user_config.training.tensorboard.port

        if not is_port_available(self.port):
            raise RuntimeError(
                f"Failed to launch TensorBoardService. Specified port {self.port} is already being used by another process."
            )

        self._process: Optional[subprocess.Popen] = None

        # register atexit shutdown
        atexit.register(self.shutdown)

        # register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)

    def __del__(self) -> None:
        """
        Destructor that ensures TensorBoardService is shut down on object deletion.
        """
        with suppress(Exception):
            self.shutdown()

    def _signal_handler(self, signum, frame) -> None:
        # Best effort cleanup, then re-raise default behavior by exiting
        try:
            self.shutdown()
        finally:
            raise SystemExit(128 + int(signum))

    @property
    def writer(self) -> SummaryWriter:
        """
        Get the internal SummaryWriter instance.

        Returns:
            SummaryWriter: The PyTorch TensorBoard SummaryWriter.
        """
        return self._writer

    def launch(self) -> Tuple[int, int]:
        """
        Launch a TensorBoard instance.

        Returns:
            (PID, Port)
        """
        with suppress_stderr():
            self._process = subprocess.Popen([
                "tensorboard",
                "--logdir", str(self.log_dir),
                "--port", str(self.port)
            ])

        logger.info(
            f"%s started with PID=%d at http://localhost:%d",
            self.__class__.__name__, self._process.pid, self.port
        )

        return self._process.pid, self.port

    def shutdown(self) -> None:
        """
        Shut down any active TensorBoardService instance if running.

        Terminates the subprocess and waits for it to exit.
        Any exception during termination is caught and logged.
        """
        with suppress(Exception):
            self._writer.close()

        proc = self._process
        if proc is None:
            return

        pid = proc.pid

        try:
            # if already exited, just clear
            if proc.poll() is not None:
                self._process = None
                logger.debug("%s already exited (PID=%d)", self.__class__.__name__, pid)
                return

            # terminate then wait briefly
            proc.terminate()
            try:
                proc.wait(timeout=3)
                logger.info("%s terminated (PID=%d, port=%d)", self.__class__.__name__, pid, self.port)
            except subprocess.TimeoutExpired:
                # escalate
                proc.kill()
                proc.wait(timeout=3)
                logger.warning("%s killed after timeout (PID=%d, port=%d)", self.__class__.__name__, pid, self.port)

        except Exception:
            logger.exception(
                "failed to shutdown %s (PID=%d, port=%d)",
                self.__class__.__name__, pid, self.port
            )
        finally:
            self._process = None