"""
Test code in src/data.py
"""
from shutil import rmtree

import numpy as np
import torch
from torchvision.datasets import MNIST

from src.data import partition_dataset


class TestPartition:
    @classmethod
    def setup_class(cls):
        cls.dataset = MNIST(".", download=True)

    @classmethod
    def teardown_class(cls):
        rmtree("MNIST")

    def test_partition_dataset_returns_disjoint_parts_of_data(self):
        dataset1, dataset2 = partition_dataset(self.dataset)

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
