import syft as sy
from __future__ import print_function
import torch
from torch.utils.data import Dataset


def split_data(n_workers, dataset)
    idx = 0
    dic_single_datasets = {}
    for i in range(0, n_workers): 
        dic_single_datasets[i] = []

    for tensor, label in dataset: 
        height = tensor.shape[-1]//n_workers
        data_parts_list = []
        #put in a list the parts to give to single workers
        for i in range(0,n_workers-1):
            dic_single_datasets[i].append(tuple([tensor[:, :, height * i : height * (i + 1)], label, idx]))
        dic_single_datasets[n_workers-1].append(tuple([tensor[:, :, height * (i+1) : ], label, idx ])) #last part of the image

        idx += 1
    #each value of the dictionary is a list of triples
    return dic_single_datasets


class BaseVerticalDataset(Dataset): 
    def __init__(self, datalist):
        self.dataset = datalist
        self.get_data_tensor()
        self.worker_id = None
        self.data_pointer = None
        self.label_pointer = None
        self.index_pointer = None
        
    def __len__(self):
        return len(self.dataset)
        
    def __get_item__(self, idx):
        return self.dataset[i]
        
    def get_data_tensor(self):
        self.data_tensor = []
        self.label_tensor = []
        self.index_tensor = []
        for el in self.dataset: 
            self.data_tensor.append(el[0])
            self.label_tensor.append(el[1])
            self.index_tensor.append(el[2])
        self.data_tensor = torch.stack(self.data_tensor)
        self.label_tensor = torch.Tensor(self.label_tensor)
        self.index_tensor = torch.Tensor(self.index_tensor)
            
    def send_to_worker(self, worker):
        self.worker_id = worker
        self.data_pointer = self.data_tensor.send(worker)
        self.label_pointer = self.label_tensor.send(worker)
        self.index_pointer = self.index_tensor.send(worker)
        return self.data_pointer, self.label_pointer, self.index_pointer


class VerticalFederatedDataset():
    #takes a list of BaseVerticalDatasets (already sent to workers)
    def __init__(self, datasets):
        
        self.datasets = {}
        
        for dataset in datasets:
            worker_id = dataset.worker_id
            self.datasets[worker_id] = dataset
    
    
    def workers(self):
        """
        Returns: list of workers
        """

        return list(self.datasets.keys())


 class VerticalFederatedDataLoader():
    
    def __init__(self, vertical_fed_dataset):
    	self.vertical_fed_dataset = vertical_fed_dataset
