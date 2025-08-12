# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from contextlib import contextmanager

__all__ = ["suppress_stderr", "suppress_output"]


@contextmanager
def suppress_stderr():
    """Context manager to suppress C-level stderr."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)  # Redirect stderr (fd 2) to /dev/null
        yield
    finally:
        os.dup2(saved_fd, 2)    # Restore original stderr
        os.close(devnull_fd)
        os.close(saved_fd)

@contextmanager
def suppress_output():
    """Suppress all C-level stdout/stderr."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Duplicate original fds
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)

    try:
        # Redirect to /dev/null
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # Restore original fds
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)
