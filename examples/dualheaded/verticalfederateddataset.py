import syft as sy
from __future__ import print_function
import torch
from torch.utils.data import Dataset


def split_data(dataset, worker_list=None, n_workers=None):
        
    if worker_list == None:
        if n_workers == None: 
            n_workers = 2  #default
        worker_list = list(range(0, n_workers))
            
    idx = 0
    
    dic_single_datasets = {}
    for worker in worker_list: 
        dic_single_datasets[worker] = [[],[],[]]
        
    for tensor, label in dataset: 
        height = tensor.shape[-1]//len(worker_list)
        i = 0
        for worker in worker_list[:-1]: 
            dic_single_datasets[worker][0].append(tensor[:, :, height * i : height * (i + 1)])
            dic_single_datasets[worker][1].append(label)
            dic_single_datasets[worker][2].append(idx)
            i += 1
            
        dic_single_datasets[worker_list[-1]][0].append(tensor[:, :, height * (i+1) : ])
        dic_single_datasets[worker_list[-1]][1].append(label)
        dic_single_datasets[worker_list[-1]][2].append(idx)
        
        idx += 1
        
    return dic_single_datasets


def split_data_create_vertical_dataset(dataset, worker_list): 
    
    dic_single_datasets = split_data(dataset, worker_list=worker_list)

    #create base datasets 
    base_datasets_list = []
    for worker in worker_list: 
        base_datasets_list.append(BaseVerticalDataset(dic_single_datasets[worker], worker_id=worker))
        
    #create VerticalFederatedDataset
    return VerticalFederatedDataset(base_datasets_list)

class BaseVerticalDataset(Dataset): 
    def __init__(self, datatuples, worker_id=None):
        
        self.fill_tensors(datatuples)
            
        self.worker_id = None
        if worker_id != None: 
            self.send_to_worker(worker_id)
            self.worker_id = worker_id
            
        self.dataset_tolist()
        
            
    def __len__(self):
        return self.data_tensor.shape[0]
        
    def __get_item__(self, idx):
        return tuple([self.data_tensor[idx], self.label_tensor[idx], self.index_tensor[idx]])
        
    def __fill_tensors(self, data_tuples):
        self.data_tensor = torch.stack(data_tuples[0])
        self.label_tensor = torch.Tensor(data_tuples[1])
        self.index_tensor = torch.Tensor(data_tuples[2])
        
    def __dataset_tolist(self):
        flat_dataset = []
        for i in range(0, self.__len__()):
            flat_dataset.append(self.__get_item__(i))
        self.dataset = flat_dataset
            
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
