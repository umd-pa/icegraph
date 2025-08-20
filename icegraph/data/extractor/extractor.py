# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import pandas as pd

from icegraph.utils.hashutils import stable_hash_cbor
from .base import Extractor

if TYPE_CHECKING:
    from icegraph.data.pipeline import Pipeline
else:
    Pipeline = None

__all__ = ["FeatureExtractor"]


class FeatureExtractor(Extractor):
    """
    Extracts features from I3 files using the IceTray module `ml_suite`,
    runs IceTray in a separate process.
    """

    _worker_proc: Optional[mp.Process] = None
    _task_q: Optional[mp.Queue] = None
    _status_q: Optional[mp.Queue] = None

    def _ensure_worker(self) -> None:
        # Safer than fork
        mp.set_start_method("spawn", force=True)

        if self._worker_proc and self._worker_proc.is_alive():
            return

        from .workers import worker_main  # import only when needed

        self._task_q = mp.Queue(maxsize=32)
        self._status_q = mp.Queue(maxsize=128)
        self._worker_proc = mp.Process(
            target=worker_main,
            args=(self._task_q, self._status_q),
            daemon=True,
        )
        self._worker_proc.start()

    def _shutdown_worker(self) -> None:
        if self._task_q:
            try:
                self._task_q.put_nowait(None)
            except Exception:
                pass
        if self._worker_proc:
            self._worker_proc.join(timeout=5)
        self._worker_proc = None
        self._task_q = None
        self._status_q = None

    def _process(self, infile: Path) -> Optional[Pipeline.Envelope]:
        # Prepare args
        frame_keys = self._config.user_config.frame_keys.toDict()
        mls_config = str(self._config.ml_suite_config_file)
        gcd_path = str(self._config.gcd_path)
        out_dir = str(self._parent.working_dir_path)

        # Launch/ensure worker
        self._ensure_worker()

        # Enqueue a job
        job_id = stable_hash_cbor({"in": str(infile), "cfg": mls_config})
        self._task_q.put({
            "job_id": job_id,
            "infile": str(infile),
            "gcd_path": gcd_path,
            "out_dir": out_dir,
            "frame_keys": frame_keys,
            "mls_config_path": mls_config,
        })

        # Wait for status 'finished' or 'error' for this job
        outfile: Optional[str] = None
        err_info: Optional[Dict] = None

        while True:
            msg = self._status_q.get()
            if msg.get("job_id") != job_id:
                continue
            if msg["status"] == "finished":
                outfile = msg["outfile"]
                break
            elif msg["status"] == "error":
                err_info = msg
                break
        if err_info:
            raise RuntimeError(
                f"IceTray worker failed on {err_info.get('infile')}: {err_info.get('error')}\n"
                f"{err_info.get('traceback')}"
            )

        # Build the env
        env = self._parent.Envelope(df=pd.DataFrame(), fh=self._parent.FileHandle(src=Path(outfile)))
        env = self._register_metadata(env, infile)
        return env

    def _register_metadata(self, env: Pipeline.Envelope, infile: Path) -> Pipeline.Envelope:
        mls_config = self._config.ml_suite_config

        env.attrs["global"]["config"] = mls_config
        env.attrs["global"]["config_hash"] = stable_hash_cbor(mls_config)
        env.attrs["origin"]["name"] = str(infile)

        return env

    def close(self) -> None:
        self._shutdown_worker()
