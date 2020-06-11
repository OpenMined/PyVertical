![om-logo](https://github.com/OpenMined/design-assets/blob/master/logos/OM/horizontal-primary-trans.png)

![Tests](https://github.com/OpenMined/PyVertical/workflows/Tests/badge.svg?branch=master)
![License](https://img.shields.io/github/license/OpenMined/PyVertical)
![OpenCollective](https://img.shields.io/opencollective/all/openmined)

# PyVertical

A project developing Privacy Preserving Vertically Distributed Learning.

- :lock: Links vertically partitioned data
         without exposing membership
         using Private Set Intersection (PSI)
- :eye: Trains a model on vertically partitioned data
        using SplitNNs,
        so only data holders can access data

## Requirements
This project is written in Python.
The work is displayed in jupyter notebooks.

To install the dependencies,
we recommend using Conda:
1. Clone this repository
1. In the command line, navigate to your local copy of the repository
1. Run `conda env create -f environment.yml`
    - This creates an environment `pyvertical-dev`
    - Comes with most dependencies you will need
1. Activate the environment with `conda activate pyvertical-dev`
1. Run `pip install syft[udacity]`
1. Run `conda install notebook`

N.b. Installing the dependencies takes several steps to circumvent versioning incompatibility between
`syft` and `jupyter`.
In the future,
all packages will be moved into the `environment.yml`.

## Usage
To create a vertically partitioned dataset:
```python
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.dataloader import VerticalDataLoader
from src.dataset import add_ids, partition_dataset

# Create dataset
data = add_ids(MNIST)(".", download=True, transform=ToTensor())  # add_ids adds unique IDs to data points

# Split data
data_partition1, data_partition2 = partition_dataset(data)

# Batch data
dataloader = VerticalDataLoader(data_partition1, batch_size=128)

for data, targets, ids in dataloader:
    # Train a model
    pass
```

## Contributing
Pull requests are welcome.
For major changes,
please open an issue first to discuss what you would like to change.

## Testing
We use [`pytest`][pytest] to test the source code.
To run the tests:
1. In the command line, navigate to the root of this repository
1. Run `python -m pytest`

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)

[conda]: https://docs.conda.io/en/latest/
[pytest]: https://docs.pytest.org/en/latest/contents.html