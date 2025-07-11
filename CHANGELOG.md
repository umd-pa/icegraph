# Changelog

### Version [0.5.0] --- July 10th, 2025:
- Added example scripts under icegraph/examples for data processing and training.
- Added installation instructions and fixed some issues with installation.
- Moved internal configuration files to icegraph.icegraph.config.defaults.
- Replaced setup.py --> MANIFEST.in, setup.cfg, pyproject.toml.
- Many changes, backend class naming has been overhauled.

### Version [0.4.0] --- July 6th, 2025:
- Restructured the API, reduced the black-boxiness of the program.
- Added file merging utilities for LMDB and HDF5.
- Aggressively refactored some portions of code, especially in icegraph.data module.
- Added GravNet model for training, along with a rudimentary Trainer class.
- Removed old trainer models and modules.
- Config is now globally accessible via IGConfig.register(), no longer need to pass downstream.
- icegraph.data.convert is now a placeholder submodule.
- Many other minor changes.

#### Version [0.3.2] --- June 22nd, 2025:
- Reorganized the config file for clarity.
- Changed the feature plot to generate 3 1D histograms instead of 1 2D histogram.

#### Version [0.3.1] --- June 21st, 2025:
- Added rudimentary feature plotting. Simply run ``plot = FeaturePlot(dataset_registry, config)``, then call ``plot.plot_feature("<feature_name>", save_path="<save_path>")``.

### Version [0.3.0] --- June 21st, 2025:
- HDF5ToParquet converter module no longer combines ID's into composite keys, leaving them as separate columns. This massively improves program speed as packing and unpacking ID's added significant overhead. This also allows for dataset splitting selection strings to target any existing column in the truth table.
- Moved project version to a config file, project name is still an IGConfig class attribute as it is not expected to change.
- DOM (x, y, z) positions are now included as parameters for training.
- Added more error handling, e.g. selection strings are now pre-verified before querying data.
- Added an option to specify the number of workers when multiprocessing under the global config file. The program will never try to start more workers than there are CPUs available.
- Some code cleanup.

#### Version [0.2.2] --- June 20th, 2025:
- Parallelized the cache builder, runs much faster now.
- Added icegraph.data.base.workers for defining/handling multiprocessing workers.
- Reordered module imports to match industry standards.

#### Version [0.2.1] --- June 20th, 2025:
- Added more error handling.
- Added data caching for fast training, need to parallelize the cache builder.
- Other changes.

### Version [0.2.0] --- June 12th, 2025:
- Restructured the project: moved icegraph submodules converter, extractor and cache to icegraph.data.
- Added some plotting functionality, can generate very basic feature plots using the icegraph.render.FeaturePlot class.
- Added icegraph.geometry submodule, added Detector class to geometry submodule to handle tasks related to the physical detector.
- Created icegraph.data.DatasetRegistry which handles generating training splits.
- Splits are resolved via a naive string resolver, allowing for selections for each split based on Event number in config.yaml. There are plans to expand this functionality.
- Moved the base IGData class to icegraph.data.base.
- Other minor fixes/changes.

#### Version [0.1.2] --- June 9th, 2025:
- Changed some class names to improve clarity.
- Other minor fixes to docs.

#### Version [0.1.1] --- June 8th, 2025:
- Added sphinx automated documentation.

### Version [0.1.0] --- June 7th, 2025:
- Full datasets can now be loaded via icegraph.dataset.Data.from_config().
- Added configuration handling via icegraph.config.Config.
- Significantly improved internal documentation.
- Condensed user configs to one file for usability, internal configs are separate.
- Slightly optimized caching for faster repeated conversions via icegraph.cache.I3ConversionCache _(since been renamed to IGConverterCache)_.
- Other minor changes.

## Version [0.0.0] --- June 6th, 2025:
- Implementation of semantic versioning. See https://semver.org/.