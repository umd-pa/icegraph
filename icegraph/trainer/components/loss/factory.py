# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import Factory

# local subpackage
from .loss import LossFunction

# implementations
from .standard import MSELoss, L1Loss, NLLLoss, CrossEntropyLoss, BCEWithLogitsLoss

__all__ = ["LossFactory"]


class LossFactory(Factory[LossFunction]):
    pass


# register each internal module
for module in [MSELoss, L1Loss, NLLLoss, CrossEntropyLoss, BCEWithLogitsLoss]:
    LossFactory.register(module)
