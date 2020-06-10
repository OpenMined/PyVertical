"""
Handling vertically partitioned data
"""
from copy import deepcopy
from typing import Tuple, TypeVar

import numpy as np


Dataset = TypeVar("Dataset")


def partition_dataset(
    dataset: Dataset, keep_order: bool = False, remove_data: bool = True,
) -> Tuple[Dataset, Dataset]:
    """
    Vertically partition a torch dataset in two

    A vertical partition is when parameters for a single data point is
    split across multiple data holders.
    This function assumes the dataset to split contains images (e.g. MNIST).
    The two parts of the split dataset are the top half and bottom half of an image.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split. Must be a dataset of images
    keep_order : bool (default = False)
        If False, shuffle the elements of each dataset
    remove_data : bool (default = True)
        If True, remove datapoints with probability 0.01

    Returns
    -------
    torch.utils.data.Dataset
        Dataset containing the first partition: the top half of the images
    torch.utils.data.Dataset
        Dataset containing the second partition: the bottom half of the images

    Raises
    ------
    AssertionError
        If the size of the provided dataset does not have three elements (i.e. is not an image dataset)
    """
    partition1 = deepcopy(dataset)
    partition2 = deepcopy(dataset)

    # Remove random subsets of data with 1% prob
    if remove_data:
        remove_idxs1 = np.random.uniform(0, 1, len(partition1)) > 0.01
        partition1.data = partition1.data[remove_idxs1]
        partition1.targets = partition1.targets[remove_idxs1]

        # Different subsets for each dataset partition
        remove_idxs2 = np.random.uniform(0, 1, len(partition2)) > 0.01
        partition2.data = partition2.data[remove_idxs2]
        partition2.targets = partition2.targets[remove_idxs2]

    data_shape = partition1.data.size()

    # Assume we're working with images at the moment
    assert len(data_shape) == 3

    half_height = int(data_shape[1] / 2)

    partition1.data = partition1.data[:, :half_height]
    partition2.data = partition2.data[:, half_height:]

    if not keep_order:
        # Shuffle dataset 1
        idxs1 = np.arange(len(partition1))
        np.random.shuffle(idxs1)

        partition1.data = partition1.data[idxs1]
        partition1.targets = partition1.targets[idxs1]

        # Shuffle dataset 2
        idxs2 = np.arange(len(partition2))
        np.random.shuffle(idxs2)

        partition2.data = partition2.data[idxs2]
        partition2.targets = partition2.targets[idxs2]

    return partition1, partition2
