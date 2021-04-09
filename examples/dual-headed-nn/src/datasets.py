import torch
from torch.utils.data import Dataset
from abc import ABC


class SampleSetWithLabelsNoIndex(Dataset):
    def __init__(self, labelset, sampleset):
        self.labelset = labelset
        self.sampleset = sampleset 
        
    def __getitem__(self, idx):
        """
        Args: 
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.sampleset[idx], self.labelset[idx]])
    
    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.values.shape[0]
    
    
class IndexSet(ABC): 
    def __init__(self,ids,owner):
        self.ids = ids
        self.owner = owner
    
    
class BaseIndexSet(Dataset, IndexSet):
    def __init__(self, ids, owner, values, is_labels=False):
        self.values_dic = {}
        for i, l in zip(ids, values):
            self.values_dic[int(i)] = l
        self.is_labels = is_labels
        self.values = torch.Tensor(values) if is_labels else torch.stack(values)
        super(BaseIndexSet, self).__init__(ids,owner)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.values[idx], self.ids[idx]])
    
    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.values.shape[0]

    
class SampleIndexSetWithLabels(Dataset, IndexSet):
    def __init__(self, labelset, sampleset, owner):
        self.labelset = labelset
        self.sampleset = sampleset 
        
        self.labels = labelset.values
        self.values = sampleset.values
        super(SampleIndexSetWithLabels, self).__init__(sampleset.ids,owner)
        
        self.values_dic = {}
        for k in labelset.values_dic.keys():
            self.values_dic[int(k)] = tuple([sampleset.values_dic[k], labelset.values_dic[k]])
                                       
    def __getitem__(self, idx):
        """
        Args: 
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        ids = super.ids()
        return tuple([self.values[idx], self.labels[idx], ids[idx]])
    
    def __len__(self):
        """
        Returns: amount of samples in the dataset
        """
        return self.values.shape[0]

    
class SampleSetPointer(Dataset):
    def __init__(self, labelset, sampleset, indexes, name=""):
        self.labels = labelset
        self.values = sampleset
        self.ids = indexes
        self.name = name
                                       
    def __getitem__(self, index):
        """
        Args: 
            idx: index of the example we want to get 
        Returns: a tuple with data, label, index of a single example.
        """
        return tuple([self.values[index], self.labels[index], self.ids[index]])
    
class VerticalFederatedDataset:
    """
    VerticalFederatedDataset, which acts as a dictionary between BaseVerticalDatasets, 
    already sent to remote workers, and the corresponding workers.
    
    Args: 
        datasets: list of IndexSet.
        n_samples: the total amount of samples we want to load
    """
    def __init__(self, datasets, n_samples=50):
        
        self.n_samples = n_samples
        
        self.datasets = {} #dictionary to keep track of IndexSet and corresponding data owners
        
        #just assuming it is the same id list for now
        indices_list = datasets[0].ids
        
        #data owner list
        self.data_owners = []
        
        for dataset in datasets:
            self.datasets[dataset.name] = dataset
            self.data_owners.append(dataset.name)
        
        #create a list of dictionaries
        self.dict_items_list = []
 
        #assuming it is sequential
        for index in range(0, n_samples):
            curr_dict = {}
            for w in self.data_owners:
                curr_dict[w] = self.datasets[w][index]
            self.dict_items_list.append(curr_dict)
            if index % 10 == 0:
                print(index)
            

    def __getitem__(self, idx):
        """
        Args:
             idx: index of the example we want to get
        Returns:
            Get dataset item from different workers
        """

        return self.dict_items_list[idx]

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        fmt_str = "FederatedDataset\n"
        return fmt_str
    
    def collate_fn(self, batch):
        return batch[0]
