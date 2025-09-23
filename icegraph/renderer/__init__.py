# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .plots import CDFPlot, PDFPlot, ChargeDistributionPlot, ParityPlot, BiasPlot, ConfusionMatrixPlot, ROCPlot

CDFPlot.__module__ = __name__
PDFPlot.__module__ = __name__
ChargeDistributionPlot.__module__ = __name__
ParityPlot.__module__ = __name__
BiasPlot.__module__ = __name__
ConfusionMatrixPlot.__module__ = __name__
ROCPlot.__module__ = __name__

__all__ = ["CDFPlot", "PDFPlot", "ChargeDistributionPlot", "ParityPlot", "BiasPlot", "ConfusionMatrixPlot", "ROCPlot"]
