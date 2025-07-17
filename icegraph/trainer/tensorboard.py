# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path
import subprocess
import atexit

from torch.utils.tensorboard import SummaryWriter

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_stderr
from icegraph.utils import is_port_available

__all__ = ["TensorBoard"]


class TensorBoard:
    """
    A utility class for managing a TensorBoard instance and writing logs.
    """

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize the TensorBoard logger.

        Args:
            log_dir (Union[str, Path]): The directory where TensorBoard logs will be stored.
        """
        self.log_dir = log_dir

        # init tensorboard
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

        # grab global config
        self._config = IGConfig.get()
        self.port = self._config.user_config.training.tensorboard.port

        self._tensorboard_proc: Optional[subprocess.Popen] = None

        # register atexit shutdown
        atexit.register(self.shutdown)

    def __del__(self) -> None:
        """
        Destructor that ensures TensorBoard is shut down on object deletion.
        """
        self.shutdown()

    @property
    def writer(self) -> SummaryWriter:
        """
        Get the internal SummaryWriter instance.

        Returns:
            SummaryWriter: The PyTorch TensorBoard SummaryWriter.
        """
        return self._writer

    def launch(self, port: Optional[int] = None) -> None:
        """
        Launch a TensorBoard instance.

        Args:
            port (Optional[int]): Localhost port to serve TensorBoard on. Defaults to value specified in config.
        """
        if port is None:
            port = self.port

        with suppress_stderr():
            self._tensorboard_proc = subprocess.Popen([
                "tensorboard",
                "--logdir", str(self.log_dir),
                "--port", str(port)
            ])
        Console.out(f"TensorBoard started with PID {self._tensorboard_proc.pid} at http://localhost:{port}")

    def shutdown(self) -> None:
        """
        Shut down any active TensorBoard instance if running.

        Terminates the subprocess and waits for it to exit.
        Any exception during termination is caught and logged.
        """
        if self._tensorboard_proc is not None:
            Console.out("Shutting down TensorBoard...")

            try:
                self._tensorboard_proc.terminate()
                self._tensorboard_proc.wait()
                self._tensorboard_proc = None
            except Exception as e:
                Console.out(f"Failed to terminate TensorBoard with PID {self._tensorboard_proc.pid}: {e}", severity=2)

            Console.out("TensorBoard shut down.")