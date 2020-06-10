"""
Test code in src/data.py
"""
from copy import deepcopy
from itertools import chain
from shutil import rmtree

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from src.data import partition_dataset


class TestPartition:
    @classmethod
    def setup_class(cls):
        cls.dataset = MNIST(".", download=True, transform=transforms.ToTensor())

    @classmethod
    def teardown_class(cls):
        rmtree("MNIST")

    def test_partition_dataset_returns_disjoint_parts_of_data(self):
        dataset1, dataset2 = partition_dataset(
            self.dataset, keep_order=True, remove_data=False
        )

        # Test that we've not lost any data
        assert (
            dataset1.data.detach().numpy().size + dataset2.data.detach().numpy().size
            == self.dataset.data.detach().numpy().size
        )

        # Recombine partitioned data
        combined_data = (
            torch.cat((dataset1.data[:1_000], dataset2.data[:1_000]), 1)
            .detach()
            .numpy()
        )

        # Test that dataset1 + dataset2 recreates entire images
        np.testing.assert_array_equal(
            combined_data, self.dataset.data[:1_000].detach().numpy()
        )

    def test_partition_jumbles_data(self):
        dataset1, dataset2 = partition_dataset(
            self.dataset, remove_data=False,
        )  # keep_order = False by default

        # Test that we've not lost any data
        assert (
            dataset1.data.detach().numpy().size + dataset2.data.detach().numpy().size
            == self.dataset.data.detach().numpy().size
        )

        # Recombine partitioned data
        combined_data = (
            torch.cat((dataset1.data[:1_000], dataset2.data[:1_000]), 1)
            .detach()
            .numpy()
        )

        # Test that dataset1 + dataset2 recreates entire images
        # Although the jumble may results in equal arrays (i.e. randomly jumble to the same state)
        # this is very unlikely to happen for 1000 datapoints
        assert (combined_data != self.dataset.data[:1_000].detach().numpy()).any()

    def test_that_data_is_shuffled_with_labels(self):
        # Create small dataset
        dataset = deepcopy(self.dataset)

        # Limit size of dataset to search
        dataset.data = dataset.data[:3]
        dataset.targets = dataset.targets[:3]

        half_data_size = int(dataset.data.shape[1] / 2)

        dataset1, dataset2 = partition_dataset(dataset, remove_data=False)

        for i in range(3):
            datum1, label1 = dataset1[i]
            datum1_original_idx = np.argmax(dataset.targets == label1)
            datum1_original, _ = dataset[datum1_original_idx]

            np.testing.assert_array_equal(
                datum1.detach().numpy(),
                datum1_original.detach().numpy()[:, :half_data_size],
            )

            datum2, label2 = dataset2[i]
            datum2_original_idx = np.argmax(dataset.targets == label2)
            datum2_original, _ = dataset[datum2_original_idx]

            np.testing.assert_array_equal(
                datum2.detach().numpy(),
                datum2_original.detach().numpy()[:, half_data_size:],
            )

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

        # Check same number of data and targets have been removed
        assert dataset1.data.size(0) == dataset1.targets.size(0)
        assert dataset2.data.size(0) == dataset2.targets.size(0)

        # Check that we haven't lost data in other dimensions
        dataset1_data_size = dataset1.data.size(1) * dataset1.data.size(2)
        dataset2_data_size = dataset2.data.size(1) * dataset2.data.size(2)
        original_dataset_data_size = self.dataset.data.size(1) * self.dataset.data.size(
            2
        )

        assert dataset1_data_size + dataset2_data_size == original_dataset_data_size

    def test_that_unique_ids_remain_attached_to_correct_data(self):
        """
        Check that IDs are not sorted differently to data/labels and are removed
        if the data is
        """
        dataset1, dataset2 = partition_dataset(self.dataset)

        # Check same number of data, targets, ids have been removed
        assert dataset1.data.size(0) == dataset1.targets.size(0) == len(dataset1.ids)
        assert dataset2.data.size(0) == dataset2.targets.size(0) == len(dataset2.ids)

        # Check that IDs are unique
        _, id1_counts = np.unique(dataset1.ids, return_counts=True)
        assert np.max(id1_counts) == 1

        _, id2_counts = np.unique(dataset2.ids, return_counts=True)
        assert np.max(id2_counts) == 1

        # Check that IDs still align with data and labels
        # Check first 100 only to save time
        for i in range(100):
            _, label = dataset1[i]
            id = dataset1.ids[i]

            id_original_index = np.where(dataset2.ids == id)[0]
            if id_original_index.size:
                # ID is in dataset2 as well, so we can compare labels
                assert dataset2.targets[id_original_index[0]] == label
