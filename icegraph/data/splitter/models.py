# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean
from typing import Union, Optional, Sequence, Iterator, ClassVar, Dict
from pathlib import Path

import pandas as pd
from icegraph.data.readers import LMDBConfiguredShardReader
from sklearn.model_selection import train_test_split

from icegraph.config import IGConfig
from icegraph.data.writers import LMDBWriter
from icegraph.console import Console
from icegraph.pathutils import PathResolver

__all__ = ["DatasetSplitter"]


class DatasetSplitter:
    """
    Splits a preprocessed LMDB dataset into train, validation, and test sets,
    and writes each split as its own LMDB sample database.

    Supports standard split strategy. Will include stratified class balancing in the future.
    """

    SPLIT_INT_MAP: ClassVar[Dict[str, int]] = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    def __init__(self, source: Union[str, Path, Sequence[Union[str, Path]]], batch_size: int = 64):
        """
        Initialize the split processor.

        Args:
            source(Union[str, Path, Sequence[Union[str, Path]]]): Path to the input file(s) (LMDB).
        """
        Console.banner("Dataset Splitter")
        self.source = source

        self._config: IGConfig = IGConfig.get()
        self.target_labels = self._config.user_config.data.target_labels
        self._table: Optional[pd.DataFrame] = None
        self.batch_size = batch_size

        # initialize writer
        self._writer: Optional[LMDBWriter] = None
        self._write_index = 0

        # initialize reader
        LMDBConfiguredShardReader.configure(source, max_open_envs=4, clean=True)
        self._reader = LMDBConfiguredShardReader()

    def __call__(self, outdir: Optional[Union[str, Path]] = None) -> Path:
        """
        Callable interface to trigger the split and write output LMDBs.

        Args:
            outdir (Optional[Union[str, Path]]): Where to save the split map LMDB.
                Defaults to a `splits/` subdirectory of the input file location.

        Returns:
            Path: Path to the split map LMDB.
        """
        return self.build_map(outdir)

    def _iter_lmdb_batches(self) ->  Iterator[pd.DataFrame]:
        """
        Reads chunked LMDB file into pandas DataFrames and yield them.

        Returns:
            Iterator[pd.DataFrame]: The deserialized table of all samples in the given chunk.
        """
        total = len(self._reader)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = self._reader[start:end]

            # unpack samples
            _, file_idxs, keys = zip(*batch)

            # build df
            df = pd.DataFrame({
                "index": list(range(start, end)),
                "file_index": file_idxs,
                "key": keys,
            })
            yield df

    def _standard_split(self, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Perform a non-stratified 60/20/20 split.

        Args:
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            pd.DataFrame
        """
        # get the seed
        seed = seed or self._config.user_config.training.seed

        # grab the local dataframe
        df = self._table.copy()

        idx_all = df.index
        train_idx, temp_idx = train_test_split(
            idx_all, test_size=0.4, random_state=seed
        )

        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed
        )

        split_int_map = type(self).SPLIT_INT_MAP

        df['split'] = split_int_map["train"]
        df.loc[val_idx, 'split'] = split_int_map["validation"]
        df.loc[test_idx, 'split'] = split_int_map["test"]

        return df[['index', 'file_index', 'key', 'split']]

    def _to_lmdb(self, table: pd.DataFrame) -> None:
        """
        Write a sample map to LMDB.

        Args:
            table (pd.DataFrame): Data to write.
        """
        self._write_index += self._writer.append(table, self._write_index)

    def build_map(self, outdir: Optional[Union[str, Path]] = None) -> Path:
        """
        Generate train/validation/test splits and save them as LMDBs.

        Args:
            outdir (Optional[Union[str, Path]]): Where to save the split map LMDB.
                Defaults to a `splits/` subdirectory of the input file location.

        Returns:
            Path: Path to the split map LMDB.
        """
        Console.out(f"Generating train/val/test splits from source databases.")

        resolver = PathResolver(path=outdir, origin=None, extension="lmdb", stage="splits")
        outfile = resolver.resolve(prefix="split_map")

        self._writer = LMDBWriter(outfile, mode="a")
        self._writer.write_metadata()

        Console.out("Running standard train/test/val split...")
        strategy = self._standard_split

        for self._table in Console.progress_bar(self._iter_lmdb_batches()):
            df_map = strategy()
            self._to_lmdb(df_map)

        Console.out(f"Split generation complete. Mapping saved to {outfile}")

        return outfile
