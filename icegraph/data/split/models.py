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

__all__ = ["SplitFactory"]


class SplitFactory:

    def __init__(self, input_file: Union[str, Path]):
        self.input_file = Path(input_file)
        self._config: IGConfig = IGConfig.get()

        # load target labels once on instantiation
        self.target_labels = self._config.user_config.data.target_labels

        self._table: Optional[pd.DataFrame] = None

    def _read_lmdb(self) -> pd.DataFrame:
        # load the input lmdb file
        env = lmdb.open(
            str(self.input_file),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        rows = []
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    unpacked = msgpack.unpackb(value, raw=False)
                    rows.append(unpacked)

        env.close()
        # Force Pandas to treat each dict as one row
        return pd.DataFrame.from_records(rows)

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self._table = self._read_lmdb()
        return self._table

    def generate_splits(self, outdir: Optional[Union[str, Path]] = None, stratify: bool = False) -> tuple[Path, Path, Path]:
        """
        Generate stratified 60/20/20 train/validation/test splits and save them as LMDB files.

        This method uses multilabel stratification (via IterativeStratification) to ensure that
        each split preserves the label distribution across all classes.

        Args:
            outdir (Optional[Union[str, Path]]): Output directory to save the LMDB files.
                Defaults to `input_file.parent / "splits"` if not specified.
            stratify (bool): If True, implements stratification. Only can be set to True if
                classifying non-continuous data. Defaults to False.

        Returns:
            tuple[Path, Path, Path]: Paths to the LMDB files for the training, validation,
            and test splits respectively (train, val, test).
        """
        Console.out(f"Generating train/val/test splits from source database: {self.input_file}")

        outdir = Path(outdir or self.input_file.parent / "splits")
        outdir.mkdir(parents=True, exist_ok=True)

        multilabel = bool(len(self.target_labels) > 1)

        if stratify:
            if multilabel:
                Console.out("Running multi-label stratification...")
                Console.spinner().start()
                df_train, df_val, df_test = self._multi_label_stratification()
            else:
                Console.out("Running single-label stratification...")
                Console.spinner().start()
                df_train, df_val, df_test = self._single_label_stratification()
        else:
            Console.out("Running standard train/test/val split...")
            Console.spinner().start()
            df_train, df_val, df_test = self._standard_split()

        Console.spinner().stop()
        Console.out("Split generation complete. Saving to LMDB...")

        # save all files
        self._to_lmdb(df_train, outdir / "train.graphs.lmdb")
        self._to_lmdb(df_val, outdir / "val.graphs.lmdb")
        self._to_lmdb(df_test, outdir / "test.graphs.lmdb")

        Console.out(f"Output files saved to {outdir}")

        return outdir / "train.graphs.lmdb", outdir / "val.graphs.lmdb", outdir / "test.graphs.lmdb"

    def _standard_split(self, seed=42):
        # First split: 60% train, 40% temp (val + test)
        df_train, df_temp = train_test_split(self.table, test_size=0.4, random_state=seed)

        # Second split: 50% of temp to val, 50% to test => 20% each of original
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    def _multi_label_stratification(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_samples = len(self.table)

        # First: 60% train, 40% temp (val+test)
        stratifier_1 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.6, 0.4])
        label_col = self.target_labels[0]
        labels = self.table[label_col].values
        train_idx, temp_idx = next(stratifier_1.split(np.zeros(n_samples), labels))

        # Second: 50/50 split of remaining 40% → 20% val, 20% test
        labels_temp = self.target_labels[temp_idx]
        stratifier_2 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.5, 0.5])
        val_sub_idx, test_sub_idx = next(stratifier_2.split(np.zeros(len(temp_idx)), labels_temp))

        # Remap val/test indices back to original df
        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        # apply to table
        df_train = self.table.iloc[train_idx].reset_index(drop=True)
        df_val = self.table.iloc[val_idx].reset_index(drop=True)
        df_test = self.table.iloc[test_idx].reset_index(drop=True)

        return df_train, df_val, df_test

    def _single_label_stratification(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_samples = len(self.table)

        # First split: 60% train, 40% temp
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        label_col = self.target_labels[0]
        labels = self.table[label_col].values
        train_idx, temp_idx = next(sss1.split(np.zeros(n_samples), labels))

        # Second split: split temp into 50/50 val/test → 20% val, 20% test
        label_col = self.target_labels[0]
        labels_temp = self.table[label_col].values[temp_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_sub_idx, test_sub_idx = next(sss2.split(np.zeros(len(temp_idx)), labels_temp))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        # Apply splits
        df_train = self.table.iloc[train_idx].reset_index(drop=True)
        df_val = self.table.iloc[val_idx].reset_index(drop=True)
        df_test = self.table.iloc[test_idx].reset_index(drop=True)

        return df_train, df_val, df_test

    def _to_lmdb(self, table: pd.DataFrame, outfile: Union[str, Path]) -> None:
        """
        Writes the given DataFrame to an LMDB file.

        Args:
            table (pd.DataFrame): Data to write.
            outfile (str): Output file path.
        """
        dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        event_id_cols = self._config.standard_id_col_config.event_id_columns

        include_cols = [c for c in table.columns if c not in dom_id_cols + event_id_cols]

        # write to lmdb
        writer = LMDBWriter(table)
        writer.write(outfile, include_cols)

