# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import socket

__all__ = ["is_port_available"]


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is available on the given host.

    Args:
        port (int): Port number to check.
        host (str): Host to bind to. Defaults to localhost "127.0.0.1".

    Returns:
        bool: True if the port is available, False if it's in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False