# Changelog

---
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
- Slightly optimized caching for faster repeated conversions via icegraph.cache.I3ConversionCache.
- Other minor changes.

### Version [0.0.0] --- June 6th, 2025:
- Implementation of semantic versioning. See https://semver.org/.
