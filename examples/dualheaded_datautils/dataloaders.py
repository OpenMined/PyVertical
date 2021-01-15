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

import datasets


"""I think this is not needed anymore"""


class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #self.collate_fn = id_collate_fn
        
        

class VerticalFederatedDataLoader(DataLoader):
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    
    
    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        self.workers = dataset.workers
        
        self.batch_samplers = {}
        for worker in self.workers:
            data_range = range(len(self.dataset))
            if shuffle:
                sampler = RandomSampler(data_range)
            else:
                sampler = SequentialSampler(data_range)
            batch_sampler = BatchSampler(sampler, self.batch_size, drop_last)
            self.batch_samplers[worker] = batch_sampler
            
        single_loaders = []
        for k in self.dataset.datasets.keys(): 
            single_loaders.append(SinglePartitionDataLoader(self.dataset.datasets[k], batch_sampler=self.batch_samplers[k]))
        
        self.single_loaders = single_loaders

    def __len__(self):
        return sum(len(x) for x in self.dataset.datasets.values()) // len(self.workers)