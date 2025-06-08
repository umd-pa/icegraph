# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import threading
import time
import atexit


__all__ = ["Spinner"]

class Spinner:
    """
    A CLI spinner for displaying ongoing background tasks in the terminal.

    Starts a background thread that prints a rotating spinner.
    """

    def __init__(self, console) -> None:
        """
        Initialize the spinner.

        Args:
            console: IceGraph console object for standard output.
        """
        self.spinner_cycle = ['|', '/', '-', '\\']
        self.delay = 0.2
        self.running = False
        self._thread = None
        self.console = console

        # Ensure spinner is stopped gracefully on program exit
        atexit.register(self.stop)

    def _spinner_task(self) -> None:
        """
        Internal method that runs in a separate thread to display the spinner animation.

        Continuously prints a rotating character until `self.running` is set to False.
        """
        idx = 0
        while self.running:
            self.console.out(
                f'{self.spinner_cycle[idx % len(self.spinner_cycle)]} Loading...',
                control_prefix="\r",
                flush=True,
                newline=False,
                include_time=False
            )
            time.sleep(self.delay)
            idx += 1
        self.console.out('Done!        ', control_prefix="\r")

    def start(self) -> None:
        """
        Start the spinner animation in a background thread.

        If the spinner is already running, this has no effect.
        """
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._spinner_task)
            self._thread.start()

    def stop(self) -> None:
        """
        Stop the spinner animation.

        Safe to call multiple times; only has an effect if the spinner is running.
        """
        if self.running:
            self.running = False
            self._thread.join()
