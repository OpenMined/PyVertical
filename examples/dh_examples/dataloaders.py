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


class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #self.collate_fn = id_collate_fn
        
class VerticalFederatedDataLoader:
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    """

    def __init__(self, vf_dataset, batch_size=8, shuffle=False, drop_last=False, *args, **kwargs):

        self.vf_dataset = vf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.workers = vf_dataset.workers
        
        self.batch_samplers = {}
        for worker in self.workers:
            data_range = range(len(list(self.vf_dataset.datasets.values())))
            if shuffle:
                sampler = RandomSampler(data_range)
            else:
                sampler = SequentialSampler(data_range)
            batch_sampler = BatchSampler(sampler, self.batch_size, drop_last)
            self.batch_samplers[worker] = batch_sampler
            
        single_loaders = []
        for k in vfd.datasets.keys(): 
            single_loaders.append(SinglePartitionDataLoader(vfd.datasets[k], batch_sampler=self.batch_samplers[k]))
        
        self.single_loaders = single_loaders
            
        
    def __iter__(self):
        return zip(*self.single_loaders)

    def __len__(self):
        return sum(len(x) for x in self.vf_dataset.datasets.values()) // len(self.workers)