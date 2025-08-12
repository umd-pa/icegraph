# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from functools import wraps

__all__ = ["disabled_class"]

def disabled_class(cls):
    """Wrapper for temporarily disabled classes. Raises an error on instantiation of the class."""
    orig_init = cls.__init__
    @wraps(orig_init)
    def new_init(self, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} is temporarily disabled and should not be used.")
    cls.__init__ = new_init
    return cls