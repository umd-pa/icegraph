# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .base import IGBasicPlot, IGDistributionPlot

IGBasicPlot.__module__ = __name__
IGDistributionPlot.__module__ = __name__

__all__ = ["IGBasicPlot", "IGDistributionPlot"]
