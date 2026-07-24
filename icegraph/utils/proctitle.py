# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import multiprocessing as mp

__all__ = ["set_proctitle"]

# module logger
import logging
logger = logging.getLogger(__name__)

try:
    from setproctitle import setproctitle as _setproctitle
except ImportError:  # optional dependency, absent on older cluster venvs
    _setproctitle = None


def set_proctitle(name: str | None = None) -> None:
    """
    Set the OS-level process title, as shown by ps/htop/top.

    Defaults to the multiprocessing process name. This is cosmetic, so a
    missing or uncooperative setproctitle never fails the caller.

    Args:
        name (str | None): Title to set. Defaults to the current process name.
    """
    if _setproctitle is None:
        logger.debug("setproctitle unavailable, process title left unchanged")
        return

    title = name if name is not None else mp.current_process().name

    try:
        _setproctitle(title)
    except Exception:
        logger.debug("could not set process title to %r", title, exc_info=True)
