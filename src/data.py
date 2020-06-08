"""
Handling vertically partitioned data
"""
from copy import deepcopy
from typing import Tuple, TypeVar


Dataset = TypeVar("Dataset")


def partition_dataset(dataset: Dataset) -> Tuple[Dataset, Dataset]:
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

    data_shape = partition1.data.size()

    # Assume we're working with images at the moment
    assert len(data_shape) == 3

    half_height = int(data_shape[1] / 2)

    partition1.data = partition1.data[:, :half_height]
    partition2.data = partition2.data[:, half_height:]

    return partition1, partition2
