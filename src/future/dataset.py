"""
Future datasets functionality.
Build from syft FederatedDatasets
"""
import logging

import syft as sy
from syft.frameworks.torch import fl
import torch

logger = logging.getLogger(__name__)


class PartitionedDataset(fl.BaseDataset):
    """
    This is a base class to be used for manipulating a partitioned dataset.
    This may be composed of a .data attribute for inputs and a .targets one for labels.
    At least one of those must be present.
    It is to be used like the MNIST Dataset object, and is useful to avoid handling
    the two inputs and label tensors separately.

    Args:
        data[list,torch tensors]: the data points, optional
        targets: Corresponding labels of the data points, optional
        transform: Function to transform the datapoints

    Raises:
        AssertionError: If data and targets are both None
    """

    def __init__(self, data=None, targets=None, transform=None, owner=None, **kwargs):
        assert not (
            data is None and targets is None
        ), "At least one of data and targets must be provided"

        self._has_data = data is not None
        self._has_targets = targets is not None

        # Call BaseDataset init
        super().__init__(data, targets, transform=transform, owner=owner, **kwargs)

    def __len__(self):
        if self.has_data:
            return len(self.data)
        else:
            return len(self.targets)

    @property
    def has_data(self):
        return self._has_data

    @property
    def has_targets(self):
        return self._has_targets

    def __getitem__(self, index):
        """
        Args:
            index[integer]: index of item to get
        Returns:
            data: Data points corresponding to the given index.
                  None if data is None
            targets: Targets correspoding to given datapoint.
                  None if targets is None
        """
        if self.has_data:
            data_elem = self.data[index]
            if self.transform_ is not None:
                # TODO: avoid passing through numpy domain
                data_elem = torch.tensor(self.transform_(data_elem.numpy()))
        else:
            data_elem = None

        if self.has_targets:
            target_elem = self.targets[index]
        else:
            target_elem = None

        return data_elem, target_elem

    def get(self):
        """
        Gets the data back from respective workers.
        """
        if self.has_data:
            self.data.get_()

        if self.has_targets:
            self.targets.get_()

        return self

    def fix_prec(self, *args, **kwargs):
        """
        Converts data of PartitionedDataset into fixed precision
        """
        if self.has_data:
            self.data.fix_prec_(*args, **kwargs)

        if self.has_targets:
            self.targets.fix_prec_(*args, **kwargs)

        return self

    def float_prec(self, *args, **kwargs):
        """
        Converts data of PartitionedDataset into float precision
        """
        if self.has_data:
            self.data.float_prec_(*args, **kwargs)

        if self.has_targets:
            self.targets.float_prec_(*args, **kwargs)

        return self

    def share(self, *args, **kwargs):
        """
        Share the data with the respective workers
        """
        if self.has_data:
            self.data.share_(*args, **kwargs)

        if self.has_targets:
            self.targets.share_(*args, **kwargs)

        return self

    def __repr__(self):
        fmt_str = "PartitionedDataset\n"

        if self.has_data:
            fmt_str += f"\tData: {self.data}\n"

        if self.has_targets:
            fmt_str += f"\ttargets: {self.targets}"

        if self.tags is not None and len(self.tags):
            fmt_str += "\n\tTags: "
            for tag in self.tags:
                fmt_str += str(tag) + " "

        if self.description is not None:
            fmt_str += (
                "\n\tDescription: " + str(self.description).split("\n")[0] + "..."
            )

        return fmt_str

    @property
    def location(self):
        """
        Get location of the data or targets
        (if data does not exist)
        """
        if self.has_data:
            return self.data.location
        else:
            return self.targets.location

    @staticmethod
    def unbufferize(worker, proto_dataset):
        """
        This method deserializes BaseDatasetPB into a PartitionedDataset.
        Args:
            proto_dataset (BaseDatasetPB): input serialized BaseDatasetPB.
        Returns:
             PartitionedDataset: deserialized BaseDatasetPB.
        """
        data = sy.serde.protobuf.serde._unbufferize(worker, proto_dataset.data)
        targets = sy.serde.protobuf.serde._unbufferize(worker, proto_dataset.targets)
        dataset_id = sy.serde.protobuf.proto.get_protobuf_id(proto_dataset.id)
        child = None
        if proto_dataset.HasField("child"):
            child = sy.serde.protobuf.serde._unbufferize(worker, proto_dataset.child)
        return PartitionedDataset(
            data=data,
            targets=targets,
            id=dataset_id,
            tags=set(proto_dataset.tags),
            description=proto_dataset.description,
            child=child,
        )


def vertically_federate(dataset, workers):
    """
    Add a method to easily transform a PartitionedDataset
    into a VerticalDataset. The dataset given is split in len(workers)
    part and sent to each workers

    Currently only supports the case where two workes are provided,
    one takes the data and the other takes the targets

    Raises:
        AssertionError: If more or fewer than two workers are provided.
            If dataset does not have both data and targets
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")

    assert len(workers) == 2, "Two workers must be provided"
    assert (
        dataset.has_data and dataset.has_targets
    ), "Dataset must have data and targets"

    datasets = []

    data = dataset.data.send(workers[0])
    datasets.append(PartitionedDataset(data=data))

    targets = dataset.targets.send(workers[1])
    datasets.append(PartitionedDataset(targets=targets))

    logger.debug("Done!")
    return VerticalDataset(datasets)


PartitionedDataset.vertically_federate = vertically_federate


class VerticalDataset(fl.FederatedDataset):
    def __init__(self, datasets):
        """This class takes a list of datasets, each of which is supposed
        to be already sent to a remote worker (they have a location), and
        acts like a dictionary based on the worker ids.
        It serves like an input for the VerticalDataLoader.

        Currently only supports the case with only two datasets -
        one has the data and the other has the targets

        Args:
            datasets (list): list of remote Datasets

        Raises:
            AssertionError: if more than 2 datasets are provided
            RuntimeError: if a dataset has neither data or targets
        """
        assert len(datasets) == 2

        self.datasets = {}
        for dataset in datasets:

            if dataset.has_data:
                worker_id = dataset.data.location.id
            elif dataset.has_targets:
                worker_id = dataset.targets.location.id
            else:
                raise RuntimeError("Dataset has neither data nor targets")

            self.datasets[worker_id] = dataset
