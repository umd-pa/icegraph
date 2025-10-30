# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Dict, List
from multiprocessing import Queue

from icegraph.exceptions import IceCubeImportError
from icegraph.console._streams import suppress_output


def _run_icetray_pipeline(
        file_batch: List[str],
        out_path: str,
        frame_keys: Dict[str, str],
        mls_config_path: str,
        corsika: bool = False
) -> None:
    """
    Run an IceTray pipeline in a child process and write extracted features to HDF5.

    Args:
        file_batch (List[str]): Ordered list of I3 files, including GCD.
        out_path (str): Destination HDF5 file path.
        frame_keys (Dict[str, str]): Mapping for frame object names (e.g., 'mctree', 'weight_dict').
        mls_config_path (str): Path to ml_suite YAML/TOML configuration.
        corsika (bool): Whether data is corsika.

    Raises:
        IceCubeImportError: If IceCube modules are unavailable in the worker environment.
    """
    # Lazy imports (child-only)
    with suppress_output():
        try:
            from icecube.icetray import I3Tray as _I3Tray
            from icecube import hdfwriter as _hdfwriter, ml_suite as _ml_suite
            from icecube.sim_services.label_events import (
                MCLabeler as _MCLabeler,
                ClassificationConverter as _ClassificationConverter
            )
        except ImportError:
            _I3Tray = IceCubeImportError.IceCubeMissingBase
            _MCLabeler = IceCubeImportError.IceCubeMissingBase
            _ClassificationConverter = IceCubeImportError.IceCubeMissingBase
            _hdfwriter = IceCubeImportError()
            _ml_suite = IceCubeImportError()

        I3Tray = _I3Tray
        MCLabeler = _MCLabeler
        ClassificationConverter = _ClassificationConverter
        hdfwriter = _hdfwriter
        ml_suite = _ml_suite

    tray = I3Tray()
    tray.Add("I3Reader", Filenamelist=file_batch)

    mclabeler_kwargs = {
        "event_properties_name": None,
        "mctree_name": frame_keys["mctree"]
    }

    if not corsika:
        mclabeler_kwargs["weight_dict_name"]            = frame_keys["weight_dict"]
        mclabeler_kwargs["bg_mctree_name"]              = frame_keys["bg_mctree"]
    else:
        mclabeler_kwargs["corsika_weight_map_name"]     = frame_keys["corsika_weight_map"]
        mclabeler_kwargs["mcpe_pid_map_name"]           = None

    # MC labels
    tray.Add(MCLabeler, **mclabeler_kwargs)

    # Feature extraction
    tray.Add(ml_suite.EventFeatureExtractorModule, cfg_file=str(mls_config_path))

    # Serialize to HDF5
    tray.AddSegment(
        hdfwriter.I3HDFWriter,
        Output=str(out_path),
        Keys=[
            "ml_suite_features",
            ("classification", ClassificationConverter()),
            "classification_emuon_entry",
            "classification_emuon_deposited",
            frame_keys["truth_dict"]
        ],
        SubEventStreams=["InIceSplit"]
    )

    with suppress_output():
        tray.Execute()

    try:
        fd = os.open(out_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def worker_main(task_q: Queue, status_q: Queue) -> None:
    """
        Consume tasks from a queue, run the IceTray pipeline, and report status.

        Args:
            task_q (Queue): Incoming tasks. Each task is a dict with:
                - 'infile' (str): Input I3 file path.
                - 'gcd_path' (str): GCD file path.
                - 'out_dir' (str): Output directory for the resulting HDF5.
                - 'frame_keys' (dict): Frame key mapping (e.g., mctree, weight_dict, truth_dict, bg_mctree).
                - 'mls_config_path' (str): ml_suite configuration path.
                - 'job_id' (str|int): Identifier used in status messages.
                Use `None` as a sentinel to stop the worker.
            status_q (Queue): Outgoing status updates. Emits dicts with:
                - 'status' ('started'|'finished'|'error'|'stopped')
                - 'job_id' (optional)
                - 'infile'/'outfile' (optional)
                - 'error'/'traceback' (on failure)

        Returns:
            None
        """
    # Keep the worker light on threads (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    while True:
        task = task_q.get()
        if task is None:
            status_q.put({"status": "stopped"})
            break

        job_id = task.get("job_id")
        try:
            infile = Path(task["infile"])
            gcd_path = str(task["gcd_path"])
            out_dir = Path(task["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / infile.with_suffix(".hdf5").name)

            file_batch = [gcd_path, str(infile)]
            frame_keys: Dict[str, str] = dict(task["frame_keys"])
            mls_config_path = str(task["mls_config_path"])
            corsika = bool(task["corsika"])

            status_q.put({"status": "started", "job_id": job_id, "infile": str(infile)})
            _run_icetray_pipeline(file_batch, out_path, frame_keys, mls_config_path, corsika)
            status_q.put({
                "status": "finished",
                "job_id": job_id,
                "infile": str(infile),
                "outfile": out_path,
            })
        except Exception as e:
            status_q.put({
                "status": "error",
                "job_id": job_id,
                "infile": task.get("infile"),
                "error": repr(e),
                "traceback": traceback.format_exc(limit=20),
            })
