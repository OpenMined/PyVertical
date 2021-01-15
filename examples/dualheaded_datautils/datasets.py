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


class BaseSet(Dataset):
    def __init__(self, ids, values, worker_id=None, is_labels=False):
        self.values_dic = {}
        for i, l in zip(ids, values):
            self.values_dic[i] = l
        self.is_labels = is_labels

        self.ids = torch.Tensor(ids)
        self.values = torch.Tensor(values) if is_labels else torch.stack(values)

        self.worker_id = None 
        if worker_id: 
            self.send_to_worker(worker_id)
        
    def send_to_worker(self, worker):
        self.worker_id = worker
        self.values_pointer = self.values.send(worker)
        self.index_pointer = self.ids.send(worker)
        return self.values_pointer, self.index_pointer
    
    def __getitem__(self, index):
        """
        Args: 
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.values[index], self.ids[index]])
    
    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.values.shape[0]

        
class SampleSetWithLabels(Dataset):
    def __init__(self, labelset, sampleset, worker_id=None):
        #TO-DO: drop non-intersecting, now just assuming they are overlapping
        #TO-DO: make sure values are sorted
        self.labelset = labelset
        self.sampleset = sampleset 
        
        self.labels = labelset.values
        self.values = sampleset.values
        self.ids = sampleset.ids 
        
        self.values_dic = {}
        for k in labelset.values_dic.keys():
            self.values_dic[k] = tuple([sampleset.values_dic[k], labelset.values_dic[k]])
                                       
        self.worker_id = None 
        if worker_id != None: 
            self.send_to_worker(worker_id)
        
    def send_to_worker(self, worker):
        self.worker_id = worker
        self.label_point, self.label_ix_pointer = self.labelset.send_to_worker(worker)
        self.value_point, self.values_ix_pointer = self.sampleset.send_to_worker(worker)
        return self.label_point, self.label_ix_pointer, self.value_point, self.values_ix_pointer
        
        
    def __getitem__(self, index):
        """
        Args: 
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.values[index], self.labels[index], self.ids[index]])

    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.values.shape[0]



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
        
        indices_list = set()
        
        #take intersecting items
        for dataset in datasets:
            indices_list.update(dataset.ids)
            self.datasets[dataset.worker_id] = dataset
            
        self.workers = self.__workers()
        
        #create a list of dictionaries
        self.dict_items_list = []
          
        for index in indices_list:
            curr_dict = {}
            for w in self.workers:
                curr_dict[w] = tuple(list(self.datasets[w].values_dic[index.item()])+[index.item()])
            
            self.dict_items_list.append(curr_dict)
            
        self.indices = list(indices_list)
                
                
    def __workers(self):
        """
        Returns: list of workers
        """
        return list(self.datasets.keys())

    def __getitem__(self, idx):
        """
        Args:
            worker_id[str,int]: ID of respective worker
        Returns:
            Get dataset item from different workers
        """

        return self.dict_items_list[idx]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):

        fmt_str = "FederatedDataset\n"
        fmt_str += f"    Distributed accross: {', '.join(str(x) for x in self.workers)}\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        return fmt_str