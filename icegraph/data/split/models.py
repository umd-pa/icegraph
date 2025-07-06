# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path
import lmdb
import msgpack

import pandas as pd
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np

from icegraph.config import IGConfig
from icegraph.data.writers import LMDBWriter
from icegraph.console import Console

__all__ = ["DatasetSplitter"]


class DatasetSplitter:
    """
    Splits a preprocessed LMDB dataset into train, validation, and test sets,
    and writes each split as its own LMDB sample database.

    Supports standard, single-label, and multi-label stratification strategies.

    This class is designed for use in a modular processing chain where input
    is a single unified LMDB file and output is a directory containing three
    separate LMDB files for training, validation, and testing.
    """

    def __init__(self, input_file: Union[str, Path]):
        """
        Initialize the split processor.

        Args:
            input_file (Union[str, Path]): Path to the input LMDB file containing all samples.
        """
        self.input_file = Path(input_file)
        self._config: IGConfig = IGConfig.get()
        self.target_labels = self._config.user_config.data.target_labels
        self._table: Optional[pd.DataFrame] = None

    def __call__(self, outdir: Optional[Union[str, Path]] = None) -> tuple[Path, Path, Path]:
        """
        Callable interface to trigger the split and write output LMDBs.

        Args:
            outdir (Optional[Union[str, Path]]): Output directory path. If None,
                defaults to `input_file.parent / "splits"`.

        Returns:
            tuple[Path, Path, Path]: Paths to train, val, and test LMDB files.
        """
        return self.generate_splits(outdir)

    def _read_lmdb(self) -> pd.DataFrame:
        """
        Reads the full LMDB file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The deserialized table of all samples.
        """
        env = lmdb.open(
            str(self.input_file),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        rows = []
        with env.begin() as txn, txn.cursor() as cursor:
            for _, value in cursor:
                rows.append(msgpack.unpackb(value, raw=False))
        env.close()
        return pd.DataFrame.from_records(rows)

    @property
    def table(self) -> pd.DataFrame:
        """
        Lazily load the LMDB data into a table on first access.

        Returns:
            pd.DataFrame: Sample table loaded from LMDB.
        """
        if self._table is None:
            self._table = self._read_lmdb()
        return self._table

    def generate_splits(self, outdir: Optional[Union[str, Path]] = None) -> tuple[Path, Path, Path]:
        """
        Generate stratified train/validation/test splits and save them as LMDBs.

        Args:
            outdir (Optional[Union[str, Path]]): Where to save the split LMDB files.
                Defaults to a `splits/` subdirectory of the input file location.

        Returns:
            tuple[Path, Path, Path]: Paths to train, validation, and test LMDBs.
        """
        Console.banner("SplitToSampleDatabases")
        Console.out(f"Generating train/val/test splits from source database: {self.input_file}")

        outdir = Path(outdir or self.input_file.parent / "splits")
        outdir.mkdir(parents=True, exist_ok=True)

        multilabel = len(self.target_labels) > 1

        if self._config.user_config.data.splits.stratify:
            Console.out(
                "Running multi-label stratification..." if multilabel else "Running single-label stratification...")
            Console.spinner().start()
            if multilabel:
                df_train, df_val, df_test = self._multi_label_stratification()
            else:
                df_train, df_val, df_test = self._single_label_stratification()
        else:
            Console.out("Running standard train/test/val split...")
            Console.spinner().start()
            df_train, df_val, df_test = self._standard_split()

        Console.spinner().stop()
        Console.out("Split generation complete. Saving to LMDB...")

        self._to_lmdb(df_train, outdir / "train.graphs.lmdb")
        self._to_lmdb(df_val, outdir / "val.graphs.lmdb")
        self._to_lmdb(df_test, outdir / "test.graphs.lmdb")

        return outdir / "train.graphs.lmdb", outdir / "val.graphs.lmdb", outdir / "test.graphs.lmdb"

    def _standard_split(self, seed: Optional[int] = None):
        """
        Perform a non-stratified 60/20/20 split.

        Args:
            seed (Optional[int]): Random seed for reproducibility.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        seed = seed or self._config.user_config.data.splits.seed
        df_train, df_temp = train_test_split(self.table, test_size=0.4, random_state=seed)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    def _multi_label_stratification(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a multi-label 60/20/20 stratified split using IterativeStratification.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        n_samples = len(self.table)
        stratifier_1 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.6, 0.4])
        labels = self.table[self.target_labels[0]].values
        train_idx, temp_idx = next(stratifier_1.split(np.zeros(n_samples), labels))

        stratifier_2 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.5, 0.5])
        labels_temp = labels[temp_idx]
        val_sub_idx, test_sub_idx = next(stratifier_2.split(np.zeros(len(temp_idx)), labels_temp))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        return (
            self.table.iloc[train_idx].reset_index(drop=True),
            self.table.iloc[val_idx].reset_index(drop=True),
            self.table.iloc[test_idx].reset_index(drop=True),
        )

    def _single_label_stratification(self, seed: Optional[int] = None) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a single-label stratified 60/20/20 split.

        Args:
            seed (Optional[int]): Random seed.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        seed = seed or self._config.user_config.data.splits.seed
        labels = self.table[self.target_labels[0]].values
        n_samples = len(labels)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        train_idx, temp_idx = next(sss1.split(np.zeros(n_samples), labels))

        labels_temp = labels[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_sub_idx, test_sub_idx = next(sss2.split(np.zeros(len(temp_idx)), labels_temp))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        return (
            self.table.iloc[train_idx].reset_index(drop=True),
            self.table.iloc[val_idx].reset_index(drop=True),
            self.table.iloc[test_idx].reset_index(drop=True),
        )

    def _to_lmdb(self, table: pd.DataFrame, outfile: Union[str, Path]) -> None:
        """
        Write a sample table to LMDB.

        Args:
            table (pd.DataFrame): Data to write.
            outfile (Union[str, Path]): Output LMDB path.
        """
        dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        event_id_cols = self._config.standard_id_col_config.event_id_columns
        include_cols = [c for c in table.columns if c not in dom_id_cols + event_id_cols]

        writer = LMDBWriter(table)
        writer.write(outfile, include_cols)


