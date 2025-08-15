# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
import os
from typing import Optional, Union, List, ClassVar, Generator, cast, Dict, Sequence
import tempfile
import multiprocessing as mp
import shutil
import queue as _queue

import pandas as pd

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_output, suppress_stderr
from .base import UniqueID, Transformer
from icegraph.pathutils import PathResolver, PathValidator
from .base.exceptions import MissingI3FilesError
from icegraph.exceptions import IceCubeImportError

import warnings

# Silence Boost.Python converter warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

# have to wrap in try/except block so sphinx can properly generate docs
try:
    from icecube.icetray import I3Tray as _I3Tray, I3Module as _I3Module
    from icecube import (
        dataclasses as _dataclasses,
        icetray as _icetray,
        dataio as _dataio,
        hdfwriter as _hdfwriter,
        ml_suite as _ml_suite
    )
    from icecube.sim_services.label_events import (
        MCLabeler as _MCLabeler,
        ClassificationConverter as _ClassificationConverter,
        MuonLabels as _MuonLabels
    )
except ImportError:
    _dataclasses = IceCubeImportError()
    _icetray = IceCubeImportError()
    _dataio = IceCubeImportError()
    _hdfwriter = IceCubeImportError()
    _ml_suite = IceCubeImportError()

    _I3Tray = IceCubeImportError.IceCubeMissingBase
    _I3Module = IceCubeImportError.IceCubeMissingBase
    _MCLabeler = IceCubeImportError.IceCubeMissingBase
    _ClassificationConverter = IceCubeImportError.IceCubeMissingBase
    _MuonLabels = IceCubeImportError.IceCubeMissingBase

dataclasses = _dataclasses
icetray = _icetray
dataio = _dataio
hdfwriter = _hdfwriter
ml_suite = _ml_suite

I3Tray = _I3Tray
I3Module = _I3Module
MCLabeler = _MCLabeler
ClassificationConverter = _ClassificationConverter
MuonLabels = _MuonLabels

__all__ = ["FeatureExtractor"]


class FeatureExtractor(Transformer):
    """
    Extracts features from I3 files using the IceTray module `ml_suite`.
    """

    cls_converter: ClassVar[Optional[ClassificationConverter]] = ClassificationConverter and ClassificationConverter()

    def __init__(self, source: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Initialize the feature extractor.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path or sequence of paths to I3 files or a directory containing I3 files.
        """
        self._config: IGConfig = IGConfig.get()

        # save source input
        self._source = source
        self._file_paths: Optional[List[Path]] = None

        # validate input gcd path
        PathValidator.is_valid_file(self._config.gcd_path)

        # Derive output directory next to the input
        resolver = PathResolver(None, origin=None, extension=None, stage="extractor")
        self.outdir = resolver.resolve(return_dir=True)

    def __call__(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        return self.extract(outfile)

    def extract(
            self,
            outfile: Optional[Union[str, Path]] = None,
            yield_in_memory: bool = False
    ) -> Union[Path, Generator[pd.DataFrame, None, None]]:
        """
        Executes the IceTray feature extraction pipeline on the input source.

        Returns:
            Path: Path to the generated HDF5 output file if yield_in_memory = False.
            Generator[pd.DataFrame, None, None]: Generator yielding output dataframes if yield_in_memory = True.
        """
        self._file_paths: List[str] = PathResolver.normalize_sources(self._source, ".i3.zst", use_str=True)
        source_repr = Console.source_repr(self._source)

        Console.banner("Feature Extractor")
        Console.out(f"Running feature extraction: {source_repr}")

        if not self._file_paths:
            raise MissingI3FilesError(f"No I3 files found in source {source_repr}")

        # Path to output file
        resolver = PathResolver(path=outfile, origin=None, extension=None, stage="extractor")
        outdir = resolver.resolve(return_dir=True)

        # grab frame keys from config
        frame_keys = self._config.user_config.frame_keys.toDict()

        # grab mls config from config
        mls_config = self._config.ml_suite_config_file

        # grab gcd path from config
        gcd_path = str(self._config.gcd_path)

        if yield_in_memory:
            # grab table names from config
            table_names = self._config.user_config.table_names.toDict()

            # return a generator object without making this method a generator function
            return self._start_child_and_monitor(
                self._file_paths,
                gcd_path,
                frame_keys,
                table_names,
                str(mls_config)
            )

        # Disk mode
        for infile in Console.progress_bar(self._file_paths):
            # get the input file list (gcd and one file)
            infiles = [gcd_path, infile]
            self._extract_to_disk(infiles, outdir, frame_keys, str(mls_config))

        Console.out(f"Output files saved to {outdir}")
        return outdir

    @classmethod
    def _extract_to_disk(cls, infiles: List[str], outdir: Path, frame_keys: Dict[str, str], mls_config: str) -> Path:
        """Runs feature extraction and saves results to disk as HDF5 files."""
        # get the output hdf5 file path
        outfile = str(outdir / Path(infiles[1]).with_suffix(".hdf5").name)
        return cls._run_tray(infiles, outfile, frame_keys, mls_config)

    @staticmethod
    def _start_child_and_monitor(
        infiles: List[str],
        gcd_path: str,
        frame_keys: Dict[str, str],
        table_names: Dict[str, str],
        mls_config: str
    ) -> Generator[pd.DataFrame, None, None]:
        """Starts a child process that streams input files and runs feature extraction, yielding the results."""
        # create a temp directory in /dev/shm for fast IO
        tmpdir_base = "/dev/shm" if os.path.isdir("/dev/shm") else None
        tmp = tempfile.TemporaryDirectory(prefix="icegraph_", dir=tmpdir_base)
        tmpdir = tmp.name

        # spawn child and initialize queue
        ctx = mp.get_context("spawn")
        queue: mp.queues.Queue = ctx.Queue(maxsize=5)

        # define and start the child process loop
        p = ctx.Process(
            target=FeatureExtractor._child_loop,
            args=(infiles, gcd_path, tmpdir, frame_keys, mls_config, queue),
            daemon=False,
        )
        p.start()

        try:
            # stream file paths from child; load, unlink, yield
            while True:
                try:
                    # try to pull from the queue
                    msg = queue.get(timeout=1.0)
                except _queue.Empty:
                    # if empty, check if the queue still exists, if not then stop monitoring
                    if not p.is_alive():
                        break
                    # loop until queue is populated
                    continue

                if msg is None:
                    # child process had indicated it is finished, stop monitoring
                    break

                h5_path = cast(str, msg)
                # silence loud and redundant mismatched header warnings
                with suppress_stderr():
                    df = cast(pd.DataFrame, pd.read_hdf(h5_path, key=table_names["features"]))
                try:
                    # attempt to unlink to free up RAM
                    os.unlink(h5_path)
                except OSError:
                    # ok if it fails, additional cleanup later
                    pass
                yield df

            p.join()
            # raise if the child fails for any reason
            if p.exitcode != 0:
                raise RuntimeError(f"FeatureExtractor child process failed (exitcode={p.exitcode}).")

        finally:
            # parent runs this, if child segfaults this will still be run
            try:
                # try to terminate queue
                queue.close()
                queue.cancel_join_thread()
            except Exception:
                pass
            try:
                if p.is_alive():
                    # terminate the child process if it still exists
                    p.terminate()
            except Exception:
                pass
            try:
                # attempt cleanup
                tmp.cleanup()
            except Exception:
                # if fails, force cleanup
                shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _child_loop(
        infiles: List[str],
        gcd_path: str,
        tmpdir: str,
        frame_keys: Dict[str, str],
        mls_config: str,
        queue: mp.queues.Queue,
    ) -> None:
        """
        Child process loop. For each input I3 file, write an HDF5 to tmpdir and
        put the path on the queue. Puts None to queue when finished.
        """
        try:
            for infile in infiles:
                base = Path(infile).with_suffix(".hdf5").name
                outfile = os.path.join(tmpdir, base)
                file_batch = [gcd_path, infile]

                FeatureExtractor._run_tray(file_batch, outfile, frame_keys, mls_config)

                # blocks if parent is behind and bounds memory usage
                queue.put(outfile)

            # Success sentinel, avoid blocking forever
            try:
                queue.put(None, timeout=1.0)
            except Exception:
                pass

        except Exception as e:
            raise ChildProcessError(f"Child process failed: {e}") from e

    @staticmethod
    def _run_tray(file_batch: List[str], outfile: str, frame_keys: Dict[str, str], mls_config: str) -> Path:
        """Runs the I3Tray pipeline that handles feature extraction via ml_suite."""
        tray = I3Tray()

        tray.Add('I3Reader', Filenamelist=file_batch)

        # This module labels MC events based on their topology
        # TODO: make this optional
        tray.Add(
            MCLabeler,
            event_properties_name=None,
            mctree_name=frame_keys["mctree"],
            weight_dict_name=frame_keys["weight_dict"],
            bg_mctree_name=frame_keys["bg_mctree"]
        )

        # This module performs the feature calculation
        tray.Add(
            ml_suite.EventFeatureExtractorModule,
            cfg_file=mls_config
        )

        tray.Add(UniqueID)

        # Serialize labels and features to HDF5
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            Output=outfile,
            Keys=[
                "ml_suite_features",
                ("classification", FeatureExtractor.cls_converter),
                "classification_emuon_entry",
                "classification_emuon_deposited",
                frame_keys["truth_dict"]
            ],
            SubEventStreams=["InIceSplit"]
        )

        with suppress_output():
            tray.Execute()

        return Path(outfile)
