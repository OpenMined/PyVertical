# Changelog

## [Unreleased]

### Added

- Dockerfile for reproducible `PyVertical` environment
- partitioned dataset extending `syft`'s `BaseDataset` for remote worker functionality
- `VerticalDataset` class for storing multiple partitioned datsets on remote workers
- Changelog

### Changed

- Moved dataset splitting helper functions to "utils" folder


## [0.1.0] - 2020-07-10

PoC release of `PyVertical` -
demonstrating the concept on MNIST.

### Added
- Helper functions for vertically partitioning MNIST dataset into datasets of images and labels
- Helper functions to randomly remove and shuffle datapoints in a dataset
- Dataloader class for loading images and labels from separate datasets
- PSI function to locally match datapoint from separate datasets
- Single-headed split neural network
- Notebook demonstrating local vertical federated learning on MNIST
- Code tested on Ubuntu 18.04, Python 3.6, 3.7, 3.8


[Unreleased]: https://github.com/OpenMined/PyVertical/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OpenMined/PyVertical/releases/tag/v0.1.0
