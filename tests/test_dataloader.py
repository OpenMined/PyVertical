"""
Test code in src/dataloader.py
"""
from shutil import rmtree

import pytest
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from src.dataloader import VerticalDataLoader, SinglePartitionDataLoader
from src.utils import add_ids, partition_dataset


class TestSinglePartitionDataset:
    @classmethod
    def setup_class(cls):
        dataset = add_ids(MNIST)(
            "./TestSinglePartitionDataset",
            download=True,
            transform=transforms.ToTensor(),
        )

        dataset1, dataset2 = partition_dataset(dataset)
        cls.dataset1 = dataset1
        cls.dataset2 = dataset2

    @classmethod
    def teardown_class(cls):
        rmtree("./TestSinglePartitionDataset")

    def test_that_vertical_dataloader_only_returns_data_which_is_not_none(self):
        dataloader1 = SinglePartitionDataLoader(self.dataset1, batch_size=100)
        for results in dataloader1:
            assert len(results) == 2

            # IDs should have been converted to string
            assert isinstance(results[1][0], str)

        dataloader2 = SinglePartitionDataLoader(self.dataset2, batch_size=100)
        for results in dataloader2:
            assert len(results) == 2

            # IDs should have been converted to string
            assert isinstance(results[1][0], str)


class TestVerticalDataLoader:
    @classmethod
    def setup_class(cls):
        cls.dataset = add_ids(MNIST)(
            "./TestVerticalDataLoader", download=True, transform=transforms.ToTensor(),
        )

    @classmethod
    def teardown_class(cls):
        rmtree("./TestVerticalDataLoader")

    def test_vertical_dataloader_batches_partitioned_datasets(self):
        dataloader = VerticalDataLoader(self.dataset, batch_size=100)

        for results in dataloader:
            assert len(results) == 2  # dataset1_data, dataset2_data

            assert len(results[0]) == 2  # images, ids1
            assert len(results[1]) == 2  # labels, ids1

            # Both IDs should be length 100
            assert len(results[0][1]) == len(results[1][1]) == 100

            # ID objects should be converted to str
            assert isinstance(results[0][1][0], str)
            assert isinstance(results[1][1][0], str)

    def test_drop_non_intersecting_removes_elements(self):
        dataloader = VerticalDataLoader(self.dataset, batch_size=100)
        sample_datapoint = dataloader.dataloader1.dataset.data[0]
        intersection = [0, 1, 2]

        dataloader.drop_non_intersecting(intersection, intersection)

        assert len(dataloader.dataloader1.dataset.data) == 3
        assert len(dataloader.dataloader1.dataset.ids) == 3
        assert len(dataloader.dataloader2.dataset.targets) == 3
        assert len(dataloader.dataloader2.dataset.ids) == 3
        assert torch.equal(sample_datapoint, dataloader.dataloader1.dataset.data[0])

    def test_drop_non_intersecting_removes_all_elements_with_empty_intersection(self):
        dataloader = VerticalDataLoader(self.dataset, batch_size=100)
        intersection = []

        dataloader.drop_non_intersecting(intersection, intersection)

        assert len(dataloader.dataloader1.dataset.data) == 0
        assert len(dataloader.dataloader1.dataset.ids) == 0
        assert len(dataloader.dataloader2.dataset.targets) == 0
        assert len(dataloader.dataloader2.dataset.ids) == 0

    def test_datasets_have_same_ids_after_drop_non_intersecting(self):
        dataloader = VerticalDataLoader(self.dataset, batch_size=128)

        intersection1 = [0, 1, 5, 10]
        ids1 = [dataloader.dataloader1.dataset.ids[i] for i in intersection1]

        intersection2 = [7, 10, 12, 1]
        ids2 = [dataloader.dataloader2.dataset.ids[i] for i in intersection2]

        dataloader.drop_non_intersecting(intersection1, intersection2)

        assert len(dataloader.dataloader1.dataset.data) == 4
        assert (dataloader.dataloader1.dataset.ids == ids1).all()

        assert len(dataloader.dataloader2.dataset.targets) == 4
        assert (dataloader.dataloader2.dataset.ids == ids2).all()
