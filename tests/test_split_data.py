"""
Test code in src/dataset.py
"""
from copy import deepcopy
from shutil import rmtree
import uuid

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from src.utils import add_ids, partition_dataset


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

    def test_get_ids_returns_list_of_strings(self):
        for id_ in self.dataset.get_ids():
            assert isinstance(id_, str)

    def test_sort_by_ids_sorts_data(self):
        dataset1, _ = partition_dataset(self.dataset)

        data_unsorted = dataset1.data.clone().numpy()
        ids1_unsorted = dataset1.get_ids()
        ids1_sorted = np.sort(ids1_unsorted)

        # Check it's not sorted to start with
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(ids1_unsorted, ids1_sorted)

        # Sort
        dataset1.sort_by_ids()
        data_after_sort = dataset1.data.clone().numpy()

        np.testing.assert_array_equal(dataset1.get_ids(), ids1_sorted)

        # Check that data has also been shuffled
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(data_after_sort, data_unsorted)

    def test_sort_by_ids_sorts_targets(self):
        _, dataset2 = partition_dataset(self.dataset)

        targets_unsorted = dataset2.targets.clone().numpy()
        ids2_unsorted = dataset2.get_ids()
        ids2_sorted = np.sort(ids2_unsorted)

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(ids2_unsorted, ids2_sorted)

        # Sort
        dataset2.sort_by_ids()
        targets_after_sort = dataset2.targets.clone().numpy()

        np.testing.assert_array_equal(dataset2.get_ids(), ids2_sorted)

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(targets_after_sort, targets_unsorted)
