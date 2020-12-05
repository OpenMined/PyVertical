import syft as sy
from __future__ import print_function
import torch
from torch.utils.data import Dataset

"""
Utility functions to split and distribute the data across different workers, 
create vertical datasets and federate them. It also contains datasets and dataloader classes. 
This code is meant to be used with dual-headed Neural Networks, where there are a bunch of different workers, 
which agrees on the labels, and there is a server  with the labels only. 

Code built upon: 
- Abbas Ismail's (@abbas5253) work on dual-headed NN. In particular, check Configuration 1: 
  https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/Configuration_1.ipynb
- Syft 2.0 Federated Learning dataset and dataloader: https://github.com/OpenMined/PySyft/tree/syft_0.2.x/syft/frameworks/torch/fl

"""


def split_data(dataset, worker_list=None, n_workers=2, label_server=None):
    """
    Utility function to create a vertical split of the data. It also creates a numerical index to keep
    track of the single data across different split. 

    Args: 
        dataset: an iterable object represent the dataset. Each element of the iterable 
        is supposed to be a tuple of [tensor, label]. It could be an iterable "Dataset" object in PyTorch. 
        #TODO: add support for taking a Dataloader as input, to iterate over batches instead of single examples.

        worker_list (optional): The list of VirtualWorkers to distribute the data vertically across. 

        n_workers(optional, default=2): The number of workers to split the data across. If worker_list is not passed, this is necessary to create the split.

        label_server (optional): the server which owns only the labels (e.g. in a dual-headed NN setting)
        #TODO: add the code to send labels to the server

    Returns: 
        a dictionary holding as keys the workers passed as parameters, or integers corresponding to the split, 
        and as values a list of lists, where the first element are the single tensor of the data, the second the labels, 
        the third the index, which is to keep track of the same data point. 
    """    
        
    if worker_list == None:
        worker_list = list(range(0, n_workers))
            
    #counter to create the index of different data samples
    idx = 0 
    
    #dictionary to accomodate the split data
    dic_single_datasets = {}
    for worker in worker_list: 
        """
        Each value is a list of three elements, to accomodate, in order: 
        - data examples (as tensors)
        - label
        - index 
        """
        dic_single_datasets[worker] = [[],[],[]] 

    """
    Loop through the dataset to split the data and labels vertically across workers. 
    Splitting method from @abbas5253: https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/distribute_data.py
    """
    for tensor, label in dataset: 
        height = tensor.shape[-1]//len(worker_list)
        i = 0
        for worker in worker_list[:-1]: 
            dic_single_datasets[worker][0].append(tensor[:, :, height * i : height * (i + 1)])
            dic_single_datasets[worker][1].append(label)
            dic_single_datasets[worker][2].append(idx)
            i += 1
            
        #add the value of the last worker / split
        dic_single_datasets[worker_list[-1]][0].append(tensor[:, :, height * (i+1) : ])
        dic_single_datasets[worker_list[-1]][1].append(label)
        dic_single_datasets[worker_list[-1]][2].append(idx)
        
        idx += 1
        
    return dic_single_datasets


def split_data_create_vertical_dataset(dataset, worker_list, label_server=None): 
    """
    Utility function to distribute the data vertically across workers and create a vertical federated dataset.

    Args: 
        dataset: an iterable object represent the dataset. Each element of the iterable 
        is supposed to be a tuple of [tensor, label]. It could be an iterable "Dataset" object in PyTorch. 
        #TODO: add support for taking a Dataloader as input, to iterate over batches instead of single examples.

        worker_list: The list of VirtualWorkers to distribute the data vertically across. 

        label_server (optional): the server which owns only the labels (e.g. in a dual-headed NN setting)

    Returns: 
        a VerticalFederatedDataset.
    """    

    #get a dictionary of workers --> list of triples (data, label, idx) representing the dataset.
    dic_single_datasets = split_data(dataset, worker_list=worker_list, label_server=label_server)

    #create base vertical datasets list, to be passed to a vertical federated dataset
    base_datasets_list = []
    for worker in worker_list: 
        base_datasets_list.append(BaseVerticalDataset(dic_single_datasets[worker], worker_id=worker))
        
    #create VerticalFederatedDataset 
    return VerticalFederatedDataset(base_datasets_list)

class BaseVerticalDataset(Dataset): 
    """
    Base Vertical Dataset class, containing a portion of a vertically splitted dataset.

    Args: 
        datatuples: a list where each element is another list (or tuple) of exactly 3 elements.
        The first one is a sample data, the second one is the corresponding label, and the third one the index
        (necessary to keep track of the same vertically splitted examples across multiple workers)

        worker_id (optional): the worker to which we want to send the dataset

    """
    def __init__(self, datatuples, worker_id=None):
        
        self._fill_tensors(datatuples)
            
        self.worker_id = None
        if worker_id != None: 
            self.send_to_worker(worker_id)
            self.worker_id = worker_id
            
        self._dataset_tolist() 
        
            
    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.data_tensor.shape[0]
        
    def __get_item__(self, idx):
        """
        Args: 
            idx: index of the example we want to get 

        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.data_tensor[idx], self.label_tensor[idx], self.index_tensor[idx]])
        
    def __fill_tensors(self, data_tuples):
        """
        Private method to fill the tensors of the tuples, labels and index. 
        """
        self.data_tensor = torch.stack(data_tuples[0])
        self.label_tensor = torch.Tensor(data_tuples[1])
        self.index_tensor = torch.Tensor(data_tuples[2])
        
    def __dataset_tolist(self):
        """
        Private method to create a compact list version of the dataset, so that len(dataset) is the number of examples. 
        """
        list_dataset = []
        for i in range(0, self.__len__()):
            list_dataset.append(self.__get_item__(i))
        self.dataset = list_dataset
            

    def send_to_worker(self, worker):
        """
        Send the dataset to a worker.

        Args: 
            the worker to which we want to send the dataset 

        Returns: 
            pointers to the remote data, the labels and the index tensors
        """
        self.worker_id = worker
        self.data_pointer = self.data_tensor.send(worker)
        self.label_pointer = self.label_tensor.send(worker)
        self.index_pointer = self.index_tensor.send(worker)
        return self.data_pointer, self.label_pointer, self.index_pointer




class VerticalFederatedDataset():
    """
    VerticalFederatedDataset, which acts as a dictionary between BaseVerticalDatasets, 
    already sent to remote workers, and the corresponding workers.
    This serves as an input to VerticalFederatedDataLoader. 
    Same principle as in Syft 2.0 for FederatedDataset: 
    https://github.com/OpenMined/PySyft/blob/syft_0.2.x/syft/frameworks/torch/fl/dataset.py

    Args: 
        datasets: list of BaseVerticalDatasets.
    """
    def __init__(self, datasets):
        
        self.datasets = {} #dictionary to keep track of BaseVerticalDatasets and corresponding workers
        
        for dataset in datasets:
            worker_id = dataset.worker_id
            self.datasets[worker_id] = dataset
    
    
    def workers(self):
        """
        Returns: list of workers
        """
        return list(self.datasets.keys())


 class VerticalFederatedDataLoader():
    """
    Data loader class, It combines a VerticalFederatedDataset and a sampler. 

    TODO: implement the class. Check also FederatedDataLoader of Syft 2.0: 
    https://github.com/OpenMined/PySyft/blob/syft_0.2.x/syft/frameworks/torch/fl/dataloader.py
    """
    
    def __init__(self, vertical_fed_dataset):
    	self.vertical_fed_dataset = vertical_fed_dataset
