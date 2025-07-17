# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any, Optional

try:
    from icecube.icetray import OMKey
    from icecube import dataio
except ImportError:
    OMKey = None
    dataio = None

from icegraph.config import IGConfig
from .exceptions import GeometryFrameNotFound
from icegraph.pathutils import PathValidator


class Detector:
    """
    Handles detector-level information, such as accessing DOM coordinates
    from the GCD geometry using a provided configuration.

    This class loads the I3Geometry frame from a GCD file and provides
    utility methods for retrieving DOM positions using string/OM/PMT info.
    """

    def __init__(self) -> None:
        """
        Initialize the Detector object with IceGraph configuration.
        """
        self._config: IGConfig = IGConfig.get()

        # initialize gcd geometry frame cache and load to memory
        self._gcd_geometry: Any | None = None
        self._load_gcd()

        # initialize dom_id cache
        self._dom_ids: Optional[list[tuple[int, int, int]]] = None

    def _load_gcd(self) -> None:
        """
        Load the I3Geometry frame from the GCD file and store the OMGeo map.

        Raises:
            GeometryFrameNotFound: If the I3Geometry frame cannot be found in the file.
        """
        gcd_file = dataio.I3File(str(self._config.gcd_path))

        geometry: Any | None = None
        # loop through frames until we find the geometry
        for frame in gcd_file:
            if "I3Geometry" in frame:
                geometry = frame["I3Geometry"]
                break

        # if geometry was not found, raise an exception
        if not geometry:
            raise GeometryFrameNotFound(f"I3Geometry does not exist in GCD file: {self._config.gcd_path}")

        self._gcd_geometry = geometry.omgeo

    def get_dom_coords(self, string: int, om: int, pmt: int) -> tuple[float, float, float]:
        """
        Get the (x, y, z) coordinates of a DOM specified by string, OM, and PMT.

        Args:
            string (int): The string number of the DOM.
            om (int): The optical module (OM) number.
            pmt (int): The PMT index (0 for standard DOMs, >=0 for mDOMs or IceCube Upgrade).

        Returns:
            tuple[float, float, float]: The (x, y, z) position of the specified DOM.
        """
        omkey = OMKey(int(string), int(om), int(pmt))
        position = self._gcd_geometry[omkey].position
        return position.x, position.y, position.z

    def get_all_dom_ids(self) -> list[tuple[int, int, int]]:
        """
        Returns a list of all DOM identifiers present in the geometry.

        Each DOM ID is a tuple: (string, om, pmt)

        Returns:
            list[tuple[int, int, int]]: List of all DOM identifiers.
        """
        if self._dom_ids is None:
            self._dom_ids = [(omkey.string, omkey.om, omkey.pmt) for omkey in self._gcd_geometry.keys()]
        return self._dom_ids
