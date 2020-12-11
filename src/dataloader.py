"""
Dataloaders for vertically partitioned data
"""
from typing import List
from typing import Tuple
from uuid import UUID

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.utils import partition_dataset


def id_collate_fn(batch: Tuple) -> List:
    """Collate data, targets and IDs  into batches

    This custom function is necessary as default collate
    functions cannot handle UUID objects.

    Args:
        batch (tuple of (data, target, id) tuples) : tuple of data returns from each index call
            to the dataset in a batch. To be turned into batched data

    Returns:
        list : List of batched data objects:
            data (torch.Tensor), targets (torch.Tensor), IDs (tuple of strings)
    """
    results = []

    for samples in zip(*batch):
        if isinstance(samples[0], UUID):
            # Turn into a tuple of strings
            samples = (*map(str, samples),)

        # Batch data
        results.append(default_collate(samples))
    return results


class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = id_collate_fn


class VerticalDataLoader:
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    i.e. the images dataset AND the labels dataset
    """

    def __init__(self, dataset, *args, **kwargs):

        # Split datasets
        self.data_partition1, self.data_partition2 = partition_dataset(
            dataset, remove_data=False, keep_order=False
        )

        assert self.data_partition1.targets is None
        assert self.data_partition2.data is None

        self.dataloader1 = SinglePartitionDataLoader(
            self.data_partition1, *args, **kwargs
        )
        self.dataloader2 = SinglePartitionDataLoader(
            self.data_partition2, *args, **kwargs
        )

    def __iter__(self):
        return zip(self.dataloader1, self.dataloader2)

    def __len__(self):
        return (len(self.dataloader1) + len(self.dataloader2)) // 2

    def drop_non_intersecting(self, intersection: List[int]):
        """Remove elements and ids in the datasets that are not in the intersection."""
        self.dataloader1.dataset.data = self.dataloader1.dataset.data[intersection]
        self.dataloader1.dataset.ids = self.dataloader1.dataset.ids[intersection]

        self.dataloader2.dataset.targets = self.dataloader2.dataset.targets[
            intersection
        ]
        self.dataloader2.dataset.ids = self.dataloader2.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        self.dataloader1.dataset.sort_by_ids()
        self.dataloader2.dataset.sort_by_ids()
