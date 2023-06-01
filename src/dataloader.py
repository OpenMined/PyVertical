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
        self.label_partition, self.data_partitions = partition_dataset(
            dataset, remove_data=False, keep_order=False
        )

        assert self.label_partition.data is None
        for dp in self.data_partitions:
            assert dp.targets is None

        self.label_dataloader = SinglePartitionDataLoader(
            self.label_partition, *args, **kwargs
        )
        self.dataloaders = []
        for dp in self.data_partitions:
            self.dataloaders.append( SinglePartitionDataLoader(
                dp, *args, **kwargs
            ))

    def __iter__(self):
        return zip(self.label_dataloader, *self.dataloaders)

    def __len__(self):
        return (len(self.label_dataloader) + sum([len(self.dataloaders[i]) for i in range(len(self.dataloaders))])) // (1 + len(self.dataloaders))

    def drop_non_intersecting(self, intersection: List[int]):
        """Remove elements and ids in the datasets that are not in the intersection."""
        self.label_dataloader.dataset.targets = self.label_dataloader.dataset.targets[intersection]
        self.label_dataloader.dataset.ids = self.label_dataloader.dataset.ids[intersection]

        for dl in self.dataloaders:
            dl.dataset.data = dl.dataset.data[intersection]
            dl.dataset.ids = dl.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        self.label_dataloader.dataset.sort_by_ids()
        for dl in self.dataloaders:
            dl.dataset.sort_by_ids()
