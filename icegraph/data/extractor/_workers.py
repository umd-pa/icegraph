# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Queue

from icegraph.console._streams import suppress_output


def _run_icetray_pipeline(
    file_batch: List[str],
    out_path: str,
    frame_keys: Dict[str, str],
    mls_config_path: str,
) -> None:
    """
    This function runs entirely in the child process.
    All heavy IceCube imports stay inside to avoid touching them in the parent.
    """
    # Lazy imports (child-only)
    from icecube.icetray import I3Tray
    from icecube import dataclasses, icetray, dataio, hdfwriter, ml_suite
    from icecube.sim_services.label_events import (
        MCLabeler,
        ClassificationConverter,
        MuonLabels,
    )

    tray = I3Tray()
    tray.Add("I3Reader", Filenamelist=file_batch)

    # MC labels
    tray.Add(
        MCLabeler,
        event_properties_name=None,
        mctree_name=frame_keys["mctree"],
        weight_dict_name=frame_keys["weight_dict"],
        bg_mctree_name=frame_keys["bg_mctree"],
    )

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
            frame_keys["truth_dict"],
        ],
        SubEventStreams=["InIceSplit"],
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
    Receive tasks, run the tray, report status.
    Expected task: dict with keys:
      - infile: str (path)
      - gcd_path: str (path)
      - out_dir: str (dir for output file)
      - frame_keys: dict
      - mls_config_path: str
      - job_id: str/int (for routing)
    Sentinel: None to exit.
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
            frame_keys = dict(task["frame_keys"])
            mls_config_path = str(task["mls_config_path"])

            status_q.put({"status": "started", "job_id": job_id, "infile": str(infile)})
            _run_icetray_pipeline(file_batch, out_path, frame_keys, mls_config_path)
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
