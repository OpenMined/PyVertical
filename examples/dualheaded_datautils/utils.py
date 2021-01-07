from __future__ import print_function
import syft as sy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple
from uuid import UUID
from uuid import uuid4
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler

import dataloaders
import datasets
from datasets import *

"""
Utility functions to split and distribute the data across different workers, 
create vertical datasets and federate them. It also contains datasets and dataloader classes. 
This code is meant to be used with dual-headed Neural Networks, where there are a bunch of different workers, 
which agrees on the labels, and there is a server  with the labels only. 
Code built upon: 
- Abbas Ismail's (@abbas5253) work on dual-headed NN. In particular, check Configuration 1: 
  https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/Configuration_1.ipynb
- Syft 2.0 Federated Learning dataset and dataloader: https://github.com/OpenMined/PySyft/tree/syft_0.2.x/syft/frameworks/torch/fl
TODO: 
    - replace ids with UUIDs
    - there is a bug in creation of BaseDataset X
    - create class for splitting the data 
    - create LabelSet and SampleSet (to accomodate later different roles of workers)
    - improve DataLoader to accomodate different sampler (e.g. random sampler when shuffle) and different batch size X
    - split function should be able to take as an input a dataloader, and not only a dataset (i.e. single sample iteration)
    - check that / modify such that it works on data different than images
    - dictionary keys should be worker ids, not workers themselves
"""




def split_data(dataset, worker_list=None, n_workers=2):
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
        dic_single_datasets[worker] = [] 

    """
    Loop through the dataset to split the data and labels vertically across workers. 
    Splitting method from @abbas5253: https://github.com/abbas5253/SplitNN-for-Vertically-Partitioned-Data/blob/master/distribute_data.py
    """
    label_list = []
    index_list = []
    for tensor, label in dataset: 
        height = tensor.shape[-1]//len(worker_list)
        i = 0
        uuid_idx = uuid4()
        for worker in worker_list[:-1]: 
            dic_single_datasets[worker].append(tensor[:, :, height * i : height * (i + 1)])
            i += 1
            
        #add the value of the last worker / split
        dic_single_datasets[worker_list[-1]].append(tensor[:, :, height * (i) : ])
        label_list.append(label)
        index_list.append(idx)
        
        idx += 1
        
    return dic_single_datasets, label_list, index_list


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

    #get a dictionary of workers --> data , label_list, index_list, ordered
    dic_single_datasets, label_list, index_list = split_data(dataset, worker_list=worker_list)
    
    #instantiate BaseSets 
    label_set = BaseSet(index_list, label_list, is_labels=True)
    base_datasets_list = []
    for w in dic_single_datasets.keys():
        bs = BaseSet(index_list, dic_single_datasets[w], is_labels=False)
        base_datasets_list.append(SampleSetWithLabels(label_set, bs, worker_id=w))
        
    #create VerticalFederatedDataset 
    return VerticalFederatedDataset(base_datasets_list)
