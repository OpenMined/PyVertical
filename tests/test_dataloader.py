"""
Test code in src/dataloader.py
"""
from shutil import rmtree

import pytest
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from src.dataloader import VerticalDataLoader, PartitionDistributingDataLoader
from src.dataset import add_ids, partition_dataset


class TestVerticalDataLoader:
    @classmethod
    def setup_class(cls):
        dataset = add_ids(MNIST)(
            "./TestVerticalDataset", download=True, transform=transforms.ToTensor(),
        )

        dataset1, dataset2 = partition_dataset(dataset)
        cls.dataset1 = dataset1
        cls.dataset2 = dataset2

    @classmethod
    def teardown_class(cls):
        rmtree("./TestVerticalDataset")

    def test_that_vertical_dataloader_only_returns_data_which_is_not_none(self):
        dataloader1 = VerticalDataLoader(self.dataset1, batch_size=100)
        for results in dataloader1:
            assert len(results) == 2

            # IDs should have been converted to string
            assert isinstance(results[1][0], str)

        dataloader2 = VerticalDataLoader(self.dataset2, batch_size=100)
        for results in dataloader2:
            assert len(results) == 2

            # IDs should have been converted to string
            assert isinstance(results[1][0], str)


class TestPartitionDistributingDataLoader:
    @classmethod
    def setup_class(cls):
        dataset = add_ids(MNIST)(
            "./TestPartitionDistributingDataLoader",
            download=True,
            transform=transforms.ToTensor(),
        )

        dataset1, dataset2 = partition_dataset(
            dataset, remove_data=False,
        )  # for now, until PSI allows us to re-balance datasets
        cls.dataset1 = dataset1
        cls.dataset2 = dataset2

    @classmethod
    def teardown_class(cls):
        rmtree("./TestPartitionDistributingDataLoader")

    def test_vertical_dataloader_batches_partitioned_datasets(self):
        dataloader = PartitionDistributingDataLoader(
            self.dataset1, self.dataset2, batch_size=100
        )

        for results in dataloader:
            assert len(results) == 2  # dataset1_data, dataset2_data

            assert len(results[0]) == 2  # images, ids1
            assert len(results[1]) == 2  # labels, ids1

            # Both IDs should be length 100
            assert len(results[0][1]) == len(results[1][1]) == 100

            # ID objects should be converted to str
            assert isinstance(results[0][1][0], str)
            assert isinstance(results[1][1][0], str)

    def test_that_dataset1_must_not_have_targets(self):
        with pytest.raises(AssertionError):
            dataloader = PartitionDistributingDataLoader(
                self.dataset2, self.dataset2, batch_size=100
            )

    def test_that_dataset2_must_not_have_data(self):
        with pytest.raises(AssertionError):
            dataloader = PartitionDistributingDataLoader(
                self.dataset1, self.dataset1, batch_size=100
            )

    def test_drop_non_intersecting_removes_correct_elements(self):
        dataloader = PartitionDistributingDataLoader(
            self.dataset1, self.dataset2, batch_size=100
        )
        sample_datapoint = dataloader.dataloader1.dataset.data[0]
        intersection = [0, 1, 2]

        dataloader.drop_non_intersecting(intersection)

        assert 3 == len(dataloader.dataloader1.dataset.data)
        assert 3 == len(dataloader.dataloader1.dataset.ids)
        assert 3 == len(dataloader.dataloader2.dataset.ids)
        assert torch.equal(sample_datapoint, dataloader.dataloader1.dataset.data[0])

    def test_drop_non_intersecting_removes_all_elements_with_empty_intersection(self):
        dataloader = PartitionDistributingDataLoader(
            self.dataset1, self.dataset2, batch_size=100
        )
        intersection = []

        dataloader.drop_non_intersecting(intersection)

        assert 0 == len(dataloader.dataloader1.dataset.data)
        assert 0 == len(dataloader.dataloader1.dataset.ids)
        assert 0 == len(dataloader.dataloader2.dataset.ids)
