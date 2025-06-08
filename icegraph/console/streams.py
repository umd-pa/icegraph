# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from contextlib import contextmanager


__all__ = ["suppress_stderr"]

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
