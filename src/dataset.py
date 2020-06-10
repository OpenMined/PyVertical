"""
Handling vertically partitioned data
"""
from copy import deepcopy
from typing import Tuple, TypeVar
from uuid import uuid4

import numpy as np


Dataset = TypeVar("Dataset")


def add_ids(cls):
    class VerticalDataset(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.ids = np.array([uuid4() for _ in range(len(self))])

        def __getitem__(self, item):
            data, target = super().__getitem__(item)
            return data, target, self.ids[item]

    return VerticalDataset


def partition_dataset(
    dataset: Dataset, keep_order: bool = False, remove_data: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Vertically partition a torch dataset in two

    A vertical partition is when parameters for a single data point is
    split across multiple data holders.
    This function assumes the dataset to split contains images (e.g. MNIST).
    The two parts of the split dataset are the top half and bottom half of an image.

    Args:
        dataset (torch.utils.data.Dataset) : The dataset to split. Must be a dataset of images, containing ids
        keep_order (bool, default = False) : If False, shuffle the elements of each dataset
        remove_data (bool, default = True) : If True, remove datapoints with probability 0.01

    Returns:
        torch.utils.data.Dataset : Dataset containing the first partition: the top half of the images
        torch.utils.data.Dataset : Dataset containing the second partition: the bottom half of the images

    Raises:
        RuntimeError : If dataset does not have an 'ids' attribute
        AssertionError : If the size of the provided dataset
            does not have three elements (i.e. is not an image dataset)
    """
    if not hasattr(dataset, "ids"):
        raise RuntimeError("Dataset does not have attribute 'ids'")

    partition1 = deepcopy(dataset)
    partition2 = deepcopy(dataset)

    # Re-index data
    idxs1 = np.arange(len(partition1))
    idxs2 = np.arange(len(partition2))

    # Remove random subsets of data with 1% prob
    if remove_data:
        idxs1 = np.random.uniform(0, 1, len(partition1)) > 0.01
        idxs2 = np.random.uniform(0, 1, len(partition2)) > 0.01

    if not keep_order:
        np.random.shuffle(idxs1)
        np.random.shuffle(idxs2)

    partition1.data = partition1.data[idxs1]
    partition1.targets = partition1.targets[idxs1]
    partition1.ids = partition1.ids[idxs1]

    partition2.data = partition2.data[idxs2]
    partition2.targets = partition2.targets[idxs2]
    partition2.ids = partition2.ids[idxs2]

    # Partition data
    data_shape = partition1.data.size()

    # Assume we're working with images at the moment
    assert len(data_shape) == 3

    half_height = int(data_shape[1] / 2)

    partition1.data = partition1.data[:, :half_height]
    partition2.data = partition2.data[:, half_height:]

    return partition1, partition2
