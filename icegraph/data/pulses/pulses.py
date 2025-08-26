# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any, Union, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from icegraph.config import IGConfig
from icegraph.utils.pathutils import PathValidator
from icegraph.exceptions import IceCubeImportError

import warnings

# Silence Boost.Python converter warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

try:
    from icecube import dataclasses as _dataclasses, dataio as _dataio, icetray as _icetray
    from icecube.icetray import OMKey as _OMKey
except ImportError:
    _dataclasses = IceCubeImportError()
    _dataio = IceCubeImportError()
    _icetray = IceCubeImportError()
    _OMKey = IceCubeImportError.IceCubeMissingBase

dataclasses = _dataclasses
dataio = _dataio
icetray = _icetray
OMKey = _OMKey


__all__ = ["Pulses"]


class Pulses:
    """
    A utility class for extracting and analyzing pulse data from an I3 file.

    Attributes:
        infile (Path): Path to the input I3 file containing pulse data.
    """

    @dataclass
    class DOMMetadata:
        run_id:             Optional[int] = None
        sub_run_id:         Optional[int] = None
        event_id:           Optional[int] = None
        sub_event_id:       Optional[int] = None
        sub_event_stream:   Optional[str] = None
        omkey:              Optional[OMKey] = None

    def __init__(self, infile: Union[str, Path], event_id: Optional[int] = None) -> None:
        """
        Initialize the Pulses object.

        Args:
            infile (Union[str, Path]): Path to the input I3 file.
            event_id (Optional[int]): The event ID to load pulses from. Defaults to the first frame.
        """
        # grab the global config instance
        self._config = IGConfig.get()

        # validate input file
        PathValidator.is_valid_file(infile)
        self.infile = Path(infile)

        self.dom_metadata = self.DOMMetadata(event_id=event_id)

        # load the pulse series
        self.pulse_series_map = self._load_pulse_series_map()

    def _load_pulse_series_map(self) -> dataclasses.I3RecoPulseSeriesMap:
        """
        Internal method to load the pulse time and charge pairs for a given DOM.

        Returns:
            pulse_map (dataclasses.I3RecoPulseSeriesMap): A map of OMKey to pulse series.
        """
        # get frame keys from config
        pulse_key = self._config.user_config.feature_extraction.pulse_key
        header_key = self._config.user_config.frame_keys.header

        # Read the file
        file = dataio.I3File(str(self.infile))
        for frame in file:
            if not all([
                frame.Has(pulse_key),
                frame.Has(header_key),
                frame.Stop == icetray.I3Frame.Physics
            ]):
                # must be a physics frame containing pulses and a header
                continue

            header = frame[header_key]

            if self.dom_metadata.event_id is not None:
                if header.event_id != self.dom_metadata.event_id:
                    continue
            else:
                self.dom_metadata.event_id = header.event_id

            self.dom_metadata.sub_run_id        = getattr(header, "sub_run_id", None)
            self.dom_metadata.run_id            = getattr(header, "run_id", None)
            self.dom_metadata.sub_event_id      = getattr(header, "sub_event_id", None)
            self.dom_metadata.sub_event_stream  = getattr(header, "sub_event_stream", None)

            pulse_map = frame[pulse_key]
            if isinstance(pulse_map, dataclasses.I3RecoPulseSeriesMapMask):
                pulse_map = pulse_map.apply(frame)

            return pulse_map

        # throw error if no frame found
        raise ValueError(
            f"No suitable Physics frame with key '{pulse_key}' and event_id '{self.dom_metadata.event_id}' found."
        )

    def _find_best_dom(self) -> OMKey:
        """
        Find the DOM with the highest total charge.

        Returns:
            OMKey: The DOM with the highest total charge.
        """
        pulse_series_map = self.pulse_series_map

        best_dom = None
        max_charge = -np.inf

        for dom, pulses in pulse_series_map.items():
            total_charge = sum(p.charge for p in pulses)
            if total_charge > max_charge and len(pulses) > 1:
                best_dom = dom
                max_charge = total_charge

        if best_dom is None:
            raise ValueError("No DOM with more than one pulse found.")

        return best_dom

    def _get_time_charge_series(self, target_dom: OMKey) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the filtered (time, charge) series.

        Args:
            target_dom (OMKey): The target DOM for which to calculate the CDF. Defaults to finding the best DOM.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - `times`: Sorted array of pulse times [ns].
                - 'charges': Corresponding array of charges.
        """
        pulses = self.pulse_series_map[target_dom]
        pulse_series = sorted(pulses, key=lambda p: p.time)

        times = np.array([p.time for p in pulse_series])
        charges = np.array([p.charge for p in pulse_series])

        return times, charges

    def _parse_target_dom(self, target_dom: Optional[OMKey]) -> OMKey:
        """
        Returns a DOM OMKey. If a DOM OMKey is passed in as an argument, simply returns that OMKey and registers it to
        the local DOM metadata dataclass.

        Args:
            target_dom (OMKey): A target DOM. If None, finds the best DOM.

        Returns:
            OMKey: The target DOM.
        """
        if target_dom is None:
            target_dom = self._find_best_dom()

        self.dom_metadata.omkey = target_dom

        return target_dom

    def get_cdf(self, target_dom: Optional[OMKey] = None) -> tuple[np.ndarray, np.ndarray, DOMMetadata]:
        """
        Compute the cumulative distribution function (CDF) of total charge over time for a given DOM.

        Args:
            target_dom (OMKey): The target DOM for which to calculate the CDF. Defaults to finding the best DOM.

        Returns:
            Tuple[np.ndarray, np.ndarray, OMKey]:
                - `times`: Sorted array of pulse times [ns].
                - `norm_cumsum_charges`: Normalized cumulative charge at each pulse time.
                - 'dom_metadata': The DOM metadata associated with the frame from which data was retrieved.
        """
        target_dom = self._parse_target_dom(target_dom)
        times, charges = self._get_time_charge_series(target_dom)

        # Compute cumulative charge
        cumsum_charge = np.cumsum(charges)
        norm_cumsum_charges = cumsum_charge / cumsum_charge[-1]

        return times, norm_cumsum_charges, self.dom_metadata

    def get_pulses(self, target_dom: Optional[OMKey] = None) -> tuple[np.ndarray, np.ndarray, DOMMetadata]:
        """
        Get charge/time arrays for a given DOM.

        Args:
            target_dom (OMKey): The target DOM for which to obtain pulse data. Defaults to finding the best DOM.

        Returns:
            Tuple[np.ndarray, np.ndarray, OMKey]:
                - `times`: Sorted array of pulse times [ns].
                - `pdf`: Charges registered by the DOM.
                - 'dom_metadata': The DOM metadata associated with the frame from which data was retrieved.
        """
        target_dom = self._parse_target_dom(target_dom)
        times, charges = self._get_time_charge_series(target_dom)

        return times, charges, self.dom_metadata
