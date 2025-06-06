# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import threading
import time
import atexit


class Spinner:
    """A spinner object for CLI outputs."""

    def __init__(self, console) -> None:
        self.spinner_cycle = ['|', '/', '-', '\\']
        self.delay = 0.2
        self.running = False
        self._thread = None
        self.console = console

        # stop the spinner on exit
        atexit.register(self.stop)

    def _spinner_task(self) -> None:
        """Handles the spinner in a separate thread as long as `self.running` is True."""
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
        """Turns on the spinner."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._spinner_task)
            self._thread.start()

    def stop(self) -> None:
        """Turns off the spinner."""
        if self.running:
            self.running = False
            self._thread.join()