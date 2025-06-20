# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean


class CacheBuildFailure(Exception):
    """
    Raised when the cache build process fails, indicating that the on-disk cache
    could not be initialized or reconstructed.
    """
    def __init__(self, message: str):
        super().__init__(message)


class InvalidCache(Exception):
    """
    Raised when the existing cache is found to be invalid or corrupted and must
    be rebuilt before use.
    """
    def __init__(self, message: str):
        super().__init__(message)


class CacheRegistrationError(Exception):
    """
    Raised when an attempt to register data in the on-disk cache fails, for
    example due to an I/O error writing feature or label files.
    """
    def __init__(self, message: str):
        super().__init__(message)
