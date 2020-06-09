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
        dataset1, dataset2 = partition_dataset(self.dataset, keep_order=True)

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
            self.dataset
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

        dataset1, dataset2 = partition_dataset(dataset)

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
