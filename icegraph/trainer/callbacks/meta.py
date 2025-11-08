# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import inspect
from abc import ABCMeta


class RequireKwargs(ABCMeta):
    def __init__(cls, name, bases, ns):
        # only enforce if the class defines its own __init__
        init = cls.__dict__.get("__init__")
        if init is not None:
            params = inspect.signature(init).parameters.values()
            has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
            if not has_varkw:
                raise TypeError(f"{cls.__qualname__}.__init__ must accept **kwargs")
        super().__init__(name, bases, ns)
