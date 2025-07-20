# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean


from typing import Type, Dict, Any
from .models import GravNet
from icegraph.trainer.base.exceptions import UnknownModelError


class ModelFactory:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, model_cls: Type) -> None:
        """Register a model class under a given key."""
        cls._registry[name] = model_cls

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Instantiate a registered model.
        Raises UnknownModelError if the name is unknown.
        """
        if name not in cls._registry:
            raise UnknownModelError(f"Model '{name}' is not registered; available: {list(cls._registry)}")
        return cls._registry[name](*args, **kwargs)


ModelFactory.register("gravnet", GravNet)
