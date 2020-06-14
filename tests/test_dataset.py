"""
Test code in src/data.py
"""
from copy import deepcopy
from shutil import rmtree
import uuid

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from src.dataloader import VerticalDataLoader
from src.dataset import add_ids, partition_dataset


class xTestVerticalDataset:
    @classmethod
    def setup_class(cls):
        cls.dataset = add_ids(MNIST)(
            "./TestVerticalDataset", download=True, transform=transforms.ToTensor()
        )

    @classmethod
    def teardown_class(cls):
        rmtree("./TestVerticalDataset")

    def test_that_ids_are_unique(self):
        assert np.unique(self.dataset.ids).size == len(self.dataset)

    def test_that_getitem_returns_id(self):
        results = self.dataset[5]

        assert len(results) == 3
        assert isinstance(results[0], torch.Tensor)  # transform should be retained

        assert isinstance(results[2], uuid.UUID)

    def test_vertical_dataset_can_be_used_in_dataloader(self):
        dataloader = VerticalDataLoader(self.dataset, batch_size=100)

        for results in dataloader:
            assert len(results) == 3
            assert len(results[2]) == 100

            # ID objects should be converted to str
            assert isinstance(results[2][0], str)

            break


class TestPartition:
    @classmethod
    def setup_class(cls):
        cls.dataset = add_ids(MNIST)(
            "./TestPartition", download=True, transform=transforms.ToTensor()
        )

    @classmethod
    def teardown_class(cls):
        rmtree("./TestPartition")

    def test_partition_dataset_returns_disjoint_parts_of_data(self):
        dataset1, dataset2 = partition_dataset(
            self.dataset, keep_order=True, remove_data=False
        )

        # ----- Test that we've not lost any data -----
        # Dataset1 has images
        assert (
            dataset1.data.detach().numpy().size
            == self.dataset.data.detach().numpy().size
        )

        # Dataset2 has labels
        # check that dataset.__len__ for label dataset still works
        assert len(dataset2) == len(self.dataset.targets)

        # ----- Test that datsets only hold either images or labels -----
        assert dataset1.targets is None
        assert dataset2.data is None

        # ----- Test that calling a dataset returns existing data + ID only -----
        data, id1 = dataset1[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(id1, uuid.UUID)

        label, id2 = dataset2[0]
        assert isinstance(label, int)
        assert isinstance(id2, uuid.UUID)

    def test_partition_shuffles_data(self):
        # Shuffle data, but don't remove any
        dataset1, dataset2 = partition_dataset(
            self.dataset, remove_data=False,
        )  # keep_order = False by default

        # ----- Test that we've not lost any data -----
        # Dataset1 has images
        assert (
            dataset1.data.detach().numpy().size
            == self.dataset.data.detach().numpy().size
        )

        # Dataset2 has labels
        assert len(dataset2.targets) == len(self.dataset.targets)

        # ----- Test that dataset1 + dataset2 recreates entire images -----
        # Although the shuffle may results in equal arrays (i.e. randomly shuffle to the same state)
        # this is very unlikely to happen for 1000 datapoints
        assert (
            dataset2.targets.detach().numpy()[:1_000]
            != self.dataset.targets.detach().numpy()[:1_000]
        ).any()

    def test_that_data_is_shuffled_with_labels(self):
        # Create small dataset
        dataset = deepcopy(self.dataset)

        # Limit size of dataset to search
        dataset.data = dataset.data[:3]
        dataset.targets = dataset.targets[:3]

        half_data_size = int(dataset.data.shape[1] / 2)

        dataset1, dataset2 = partition_dataset(dataset, remove_data=False)

        for i in range(3):
            datum1, id1 = dataset1[i]
            datum1_original_idx = np.argmax(dataset.ids == id1)
            datum1_original, _, id1_original = dataset[datum1_original_idx]

            np.testing.assert_array_equal(
                datum1.detach().numpy(), datum1_original.detach().numpy(),
            )

            label2, id2 = dataset2[i]
            datum2_original_idx = np.argmax(dataset.ids == id2)
            _, datum2_original_label, id2_original = dataset[datum2_original_idx]

            assert label2 == datum2_original_label

    def test_that_partition_removes_data(self):
        """
        Default behaviour of partition_dataset should be to remove data
        """
        dataset1, dataset2 = partition_dataset(self.dataset)

        # Original dataset had 60_000 data points
        # with 1% prob of removal (drawn from uniform dist.) we should expect
        # between 59_200 and 59_600 data points greater than >99.99% of the time
        assert 59_200 <= len(dataset1) <= 59_600
        assert 59_200 <= len(dataset2) <= 59_600

        assert len(dataset1) == len(dataset1.ids)
        assert len(dataset2) == len(dataset2.ids)