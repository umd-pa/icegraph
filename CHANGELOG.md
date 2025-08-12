# Changelog

### Version [0.7.0] --- August 12th, 2025:
- Fixed a bug with normalization in `FeatureProcessor`. Normalization now happens at runtime on the accelerator.
- Data is now stored in LMDB under the 'data' sub-database, added a new 'meta' sub-database which stores local sample statistics, schemas, and other info.
- Each file now stores local statistics, allowing for global statistics generation at runtime. This allows for on-the-fly changing to normalization schemes, and if a file is lost normalization is automatically modified to account for it without having to reprocess files.
- Added `DatasetRegistry.profile()` which allows measurement of data throughput speeds. This helps with dataloader tuning.
- Added the base class `icegraph.trainer.callbacks.NormCallback`, which allows for creation of custom normalizers for training.
- Packaged normalizers can be selected in config.yaml under `training:normalizer:`.
- Modified example scripts and README to reflect API changes.
- `LMDBMerger` is marked as disabled until a future fix.
- Added `icegraph.data.readers.LMDBConfiguredShardReader` which allows for very efficient, high-speed, and multiprocess safe reading of any number of LMDB files. API is the same for any number of files. This makes file merging all but unnecessary, and circumvents memory constraints by only holding a set number of environments open at one time. Must be pre-configured via `LMDBConfiguredShardReader.configure()`.
- Added `icegraph.data.readers.LMDBReader` for simple LMDB file reads.
- Suppressed some redundant warnings that cluttered CLI.
- Renamed `DatasetSplitter` to `SplitMapBuilder` to reflect its change in functionality. Now generates a map file containing split info instead of splitting files on disk.
- Removed stratified splitting until a future update.
- `IGData` must now be configured via `IGData.configure()` before instantiation of datasets.
- `IGData` now inherits from `torch.utils.data.Dataset` instead of `torch_geometric.data.Dataset`.
- Features are now stored in LMDB as dense arrays, which allows for much faster load speeds and better manipulation. Column names are stored under the metadata sub-database`meta/schema:`.
- Most processes now accept sources as inputs, which can be a single file, a list of files, or a directory containing files.
- Added `icegraph.utils.Statistics`, which handles calculation of statistics and stat merging.
- Began building tests, very minimal at the moment.
- Some API changes (see README.md) and bug fixes.

#### Version [0.6.2] --- July 31st, 2025
- Fixed bug where if run outside of Icetray environment, got a cryptic attribute error. Now raises a descriptive import error.
- Added `CDFPlot`, `PDFPlot`, `ChargeDistPlot`, and a new base class `IGDistributionPlot`.
- Added a new subpackage `icegraph.data.pulses` containing a `Pulses` type that holds some utility methods for pulse data analysis.

#### Version [0.6.1] --- July 25th, 2025
- Fixed a bug with multi-objective regression resulting in mismatched tensor sizes.
- Improved the `LMDBMerger`, now runs around 3x faster.

### Version [0.6.0] --- July 19th, 2025:
- Refactored the `Trainer` for more future extensibility. Created a `ModelFactory` and reworked the `Trainer` to run on a callback architecture.
- Fixed an issue with incorrect RMSE and MSE calculation during training. Both are now correct.
- Fixed the autodocs, this time they work as expected.
- Added more robust error handling throughout. Still work in progress for full error handling.
- Updated installation instructions in README.md.
- Improved and pruned the config system slightly, and added config validation via pydantic.
- Added TensorBoard support via the `TensorBoardCallback` callback.

#### Version [0.5.3] --- July 13th, 2025:
- Slightly renamed some `icegraph.data` submodules for clarity.
- Renamed `TransformToDataset` -> `FeatureProcessor`.
- Added `icegraph.pathutils` for more user friendly path handling.
- `Trainer` now runs test and saves model after each train epoch.
- Added `default_dir` to config.yaml; if no path is passed to any part of the pipeline, the system will automatically store and organize files here.
- Modified examples scripts to reflect API changes.
- Removed program_metadata.yaml, it was completely pointless. Replaced with `icegraph.__version__`.
- Another attempt to fix autodocs.

#### Version [0.5.2] --- July 11th, 2025:
- (Hopefully) fixed auto documentation. _(Didn't work :(, will fix later)_

#### Version [0.5.1] --- July 11th, 2025:
- Added a minimal usage guide for non-parallelized workflows.

### Version [0.5.0] --- July 10th, 2025:
- Added example scripts under icegraph/examples for data processing and training.
- Added installation instructions and fixed some issues with installation.
- Moved internal configuration files to `icegraph.icegraph.config.defaults`.
- Replaced setup.py --> MANIFEST.in, setup.cfg, pyproject.toml.
- Many changes, backend class naming has been overhauled.

### Version [0.4.0] --- July 6th, 2025:
- Restructured the API, reduced the black-boxiness of the program.
- Added file merging utilities for LMDB and HDF5.
- Aggressively refactored some portions of code, especially in `icegraph.data` module.
- Added `GravNet` model for training, along with a rudimentary `Trainer` class.
- Removed old trainer models and modules.
- Config is now globally accessible via `IGConfig.register()`, no longer need to pass downstream.
- `icegraph.data.convert` is now a placeholder submodule.
- Many other minor changes.

#### Version [0.3.2] --- June 22nd, 2025:
- Reorganized the config file for clarity.
- Changed the feature plot to generate 3 1D histograms instead of 1 2D histogram.

#### Version [0.3.1] --- June 21st, 2025:
- Added rudimentary feature plotting. Simply run ``plot = FeaturePlot(dataset_registry, config)``, then call ``plot.plot_feature("<feature_name>", save_path="<save_path>")``.

### Version [0.3.0] --- June 21st, 2025:
- `HDF5ToParquet` _(since been replaced with FeatureProcessor)_ converter module no longer combines ID's into composite keys, leaving them as separate columns. This massively improves program speed as packing and unpacking ID's added significant overhead. This also allows for dataset splitting selection strings to target any existing column in the truth table.
- Moved project version to a config file, project name is still an `IGConfig` class attribute as it is not expected to change.
- DOM (x, y, z) positions are now included as parameters for training.
- Added more error handling, e.g. selection strings are now pre-verified before querying data.
- Added an option to specify the number of workers when multiprocessing under the global config file. The program will never try to start more workers than there are CPUs available.
- Some code cleanup.

#### Version [0.2.2] --- June 20th, 2025:
- Parallelized the cache builder, runs much faster now.
- Added `icegraph.data.base.workers` for defining/handling multiprocessing workers.
- Reordered module imports to match industry standards.

#### Version [0.2.1] --- June 20th, 2025:
- Added more error handling.
- Added data caching for fast training, need to parallelize the cache builder.
- Other changes.

### Version [0.2.0] --- June 12th, 2025:
- Restructured the project: moved icegraph submodules converter, extractor and cache to `icegraph.data`.
- Added some plotting functionality, can generate very basic feature plots using the `icegraph.render.FeaturePlot` class.
- Added `icegraph.geometry` submodule, added `Detector` class to geometry submodule to handle tasks related to the physical detector.
- Created `icegraph.data.DatasetRegistry` which handles generating training splits.
- Splits are resolved via a naive string resolver, allowing for selections for each split based on Event number in config.yaml. There are plans to expand this functionality.
- Moved the base `IGData` class to `icegraph.data.base`.
- Other minor fixes/changes.

#### Version [0.1.2] --- June 9th, 2025:
- Changed some class names to improve clarity.
- Other minor fixes to docs.

#### Version [0.1.1] --- June 8th, 2025:
- Added sphinx automated documentation.

### Version [0.1.0] --- June 7th, 2025:
- Full datasets can now be loaded via `icegraph.dataset.Data.from_config()`.
- Added configuration handling via icegraph.config.Config.
- Significantly improved internal documentation.
- Condensed user configs to one file for usability, internal configs are separate.
- Slightly optimized caching for faster repeated conversions via `icegraph.cache.I3ConversionCache` _(since been renamed to IGConverterCache)_.
- Other minor changes.

## Version [0.0.0] --- June 6th, 2025:
- Implementation of semantic versioning. See https://semver.org/.