import pytest
import torch as th
import syft as sy

from src.future import PartitionedDataset, VerticalDataset


hook = sy.TorchHook(th)


def test_partitioned_dataset_with_data_and_targets():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = PartitionedDataset(inputs, targets)

    assert dataset.has_data == True
    assert dataset.has_targets == True

    assert len(dataset) == 4
    assert dataset[2] == (3, 3)

    dataset_pointer = dataset.send(alice)
    assert dataset_pointer.data.location.id == "alice"
    assert dataset_pointer.targets.location.id == "alice"
    assert dataset_pointer.location.id == "alice"

    dataset = dataset_pointer.get()
    with pytest.raises(AttributeError):
        assert dataset.data.location.id == 0
    with pytest.raises(AttributeError):
        assert dataset.targets.location.id == 0

    alice.remove_worker_from_local_worker_registry()


def test_partitioned_dataset_with_data_only():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0])
    dataset = PartitionedDataset(data=inputs)

    assert dataset.has_data == True
    assert dataset.has_targets == False

    assert len(dataset) == 4
    assert dataset[2] == (3, None)

    dataset_pointer = dataset.send(alice)
    assert dataset_pointer.data.location.id == "alice"
    assert dataset_pointer.location.id == "alice"

    # Targets (None) should still be sent
    # TODO should it?
    assert dataset_pointer.targets.location.id == "alice"

    dataset = dataset_pointer.get()
    with pytest.raises(AttributeError):
        assert dataset.data.location.id == 0

    alice.remove_worker_from_local_worker_registry()


def test_partitioned_dataset_with_targets_only():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)

    targets = th.tensor([1, 2, 3, 4.0])
    dataset = PartitionedDataset(targets=targets)

    assert dataset.has_data == False
    assert dataset.has_targets == True

    assert len(dataset) == 4
    assert dataset[2] == (None, 3)

    dataset_pointer = dataset.send(alice)
    assert dataset_pointer.targets.location.id == "alice"
    assert dataset_pointer.location.id == "alice"

    # Data should still be sent
    # TODO should it?
    assert dataset_pointer.data.location.id == "alice"

    dataset = dataset_pointer.get()
    with pytest.raises(AttributeError):
        assert dataset.targets.location.id == 0

    alice.remove_worker_from_local_worker_registry()


def test_partitioned_dataset_transform():
    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    transform_dataset = PartitionedDataset(inputs, targets)

    def func(x):
        return x * 2

    transform_dataset.transform(func)

    expected_val = th.tensor([2, 4, 6, 8])
    transformed_val = [val[0].item() for val in transform_dataset]

    assert expected_val.equal(th.tensor(transformed_val).long())


def test_partitioned_dataset_transform_with_no_data():
    targets = th.tensor([1, 2, 3, 4.0])

    transform_dataset = PartitionedDataset(targets=targets)

    def func(x):
        return x * 2

    with pytest.raises(TypeError):
        transform_dataset.transform(func)


def test_vertically_federated_raises_if_more_than_two_workers_are_provided():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
    charlie = sy.VirtualWorker(id="charlie", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    dataset = PartitionedDataset(inputs, targets)

    with pytest.raises(AssertionError):
        vertical_dataset = dataset.vertically_federate((alice, bob, charlie))

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()
    charlie.remove_worker_from_local_worker_registry()


def test_vertically_federate_raises_if_dataset_does_not_have_data_and_targets():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    data_only_dataset = PartitionedDataset(data=inputs)

    with pytest.raises(AssertionError):
        vertical_dataset = data_only_dataset.vertically_federate((alice, bob))

    targets_only_dataset = PartitionedDataset(targets=targets)

    with pytest.raises(AssertionError):
        vertical_dataset = targets_only_dataset.vertically_federate((alice, bob))

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()


def test_vertically_federate_returns_a_vertical_dataset():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0]).tag("#test")
    targets = th.tensor([1, 2, 3, 4.0]).tag("#test")

    datset = PartitionedDataset(data=inputs, targets=targets)

    vertical_dataset = datset.vertically_federate((alice, bob))

    assert isinstance(vertical_dataset, VerticalDataset)

    assert vertical_dataset.workers == ["alice", "bob"]

    alice_results = alice.search(["#test"])
    assert len(alice_results) == 1

    bob_results = bob.search(["#test"])
    assert len(bob_results) == 1

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()


def test_that_vertical_dataset_can_return_datsets():
    sy.local_worker.clear_objects()

    alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
    bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    datset = PartitionedDataset(data=inputs, targets=targets)

    vertical_dataset = datset.vertically_federate((alice, bob))
    assert vertical_dataset.workers == ["alice", "bob"]

    # Collect alice's dataset
    alice_dataset = vertical_dataset.get_dataset("alice")

    # VerticalDataset should only have bob now
    assert vertical_dataset.workers == ["bob"]

    alice.remove_worker_from_local_worker_registry()
    bob.remove_worker_from_local_worker_registry()
